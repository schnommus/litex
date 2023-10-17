import unittest
import random

from migen import *
from migen.fhdl import verilog

from litex.gen.sim import *
from litex.soc.interconnect.stream import *

def float_to_fp(v, dw=32, fbits=16):
    if v >= 0:
        return int(v * (1 << fbits)) & (2**dw-1)
    else:
        return (int(-v * (1 << fbits)) ^ (2**dw-1))+1

def fp_to_float(v, dw=32, fbits=16):
    if not v & 0x80000000:
        return float(v) / (1 << fbits)
    else:
        return -float(~(v-1) & 0xffffffff) / (1<<fbits)


class FixMac(CombinatorialActor):
    """ Compute a * b + c => z. """
    def __init__(self, dw=32, fbits=16):
        dtype = (dw, True) # Signed dw-wide signals.
        self.sink = Endpoint([("a", dtype), ("b", dtype), ("c", dtype)])
        self.source = Endpoint([("z", dtype)])
        CombinatorialActor.__init__(self)
        self.comb += [
            # WARN: simulation here widens to 64, should synth?
            self.source.payload.z.eq(
                (((self.sink.payload.a * self.sink.payload.b) >> fbits) +
                    self.sink.payload.c)
            )
        ]

class RRMux(Module):
    def __init__(self, n, inner):
        self.submodules.inner = inner
        self.submodules.mux = mux = Multiplexer(
                layout=inner.sink.description.payload_layout, n=n)
        self.submodules.demux = demux = Demultiplexer(
                layout=inner.source.description.payload_layout, n=n)

        self.sel = Signal(max=n)

        self.comb += [
            mux.source.connect(inner.sink),
            inner.source.connect(demux.sink),
            mux.sel.eq(self.sel),
            demux.sel.eq(self.sel),
        ]

        self.sync += [
            If(self.sel != n,
               self.sel.eq(self.sel + 1)
            ).Else(
               self.sel.eq(0)
            )
        ]

        self.ports_allocated = 0
        self.max_ports = n

    def get_port(self):
        port_sink = getattr(self.mux, "sink"+str(self.ports_allocated))
        port_source = getattr(self.demux, "source"+str(self.ports_allocated))
        self.ports_allocated += 1
        if self.ports_allocated > self.max_ports:
            raise ValueError
        return (port_sink, port_source)


class DcBlock(Module):
    def __init__(self, mac, dw=32, fbits=16):
        dtype = (dw, True) # signed, dw-wide
        self.sink = Endpoint([("sample", dtype)])
        self.source = Endpoint([("sample", dtype)])
        self.mac_sink, self.mac_source = mac.get_port()

        self.x_k = Signal(dtype)
        self.x_k1 = Signal(dtype)
        self.y_k1 = Signal(dtype)

        fsm = FSM(reset_state="WAIT-SINK-VALID")
        self.submodules += fsm

        fsm.act("WAIT-SINK-VALID",
            self.sink.ready.eq(1),
            self.source.valid.eq(0),
            If(self.sink.valid,
               NextValue(self.x_k, self.sink.payload.sample),
               NextState("WAIT-MAC-VALID"),
            )
        )
        fsm.act("WAIT-MAC-VALID",
            self.sink.ready.eq(0),
            self.source.valid.eq(0),
            self.mac_sink.payload.a.eq(self.y_k1),
            self.mac_sink.payload.b.eq(float_to_fp(0.9875)),
            self.mac_sink.payload.c.eq(self.x_k - self.x_k1),
            self.mac_sink.valid.eq(1),
            self.mac_source.ready.eq(1),
            If(self.mac_source.valid,
               NextValue(self.y_k1, self.mac_source.payload.z),
               NextValue(self.x_k1, self.x_k),
               NextState("WAIT-SOURCE-READY"),
            )
        )
        fsm.act("WAIT-SOURCE-READY",
            self.sink.ready.eq(0),
            self.source.valid.eq(1),
            self.source.payload.sample.eq(self.y_k1),
            If(self.source.ready,
               NextValue(self.source.valid, 0),
               NextState("WAIT-SINK-VALID"),
            )
        )

class LadderLpf(Module):
    def __init__(self, mac, dw=32, fbits=16):
        dtype = (dw, True) # signed, dw-wide
        self.sink = Endpoint([("sample", dtype)])
        self.source = Endpoint([("sample", dtype)])
        self.mac_sink, self.mac_source = mac.get_port()

        # Tweakable from outside this core.
        self.g          = Signal(dtype) # filter cutoff
        self.resonance  = Signal(dtype) # filter resonance

        # Registers used throughout state transitions
        self.x          = Signal(dtype)
        self.rezz       = Signal(dtype)
        self.sat        = Signal(dtype)
        self.a1         = Signal(dtype)
        self.a2         = Signal(dtype)
        self.a3         = Signal(dtype)
        self.y          = Signal(dtype)

        # Saturation thresholds
        SAT_HI = Constant(float_to_fp(1.0), dtype)
        SAT_LO = -Constant(float_to_fp(1.0), dtype)

        fsm = FSM(reset_state="WAIT-SINK-VALID")
        self.submodules += fsm

        def fsm_mac(this_state, next_state,
                    a, b, c, z):
            """Construct a state to pend an a*b+c MAC and transition once done."""
            fsm.act(this_state,
                self.mac_sink.payload.a.eq(a),
                self.mac_sink.payload.b.eq(b),
                self.mac_sink.payload.c.eq(c),
                self.mac_sink.valid.eq(1),
                self.mac_source.ready.eq(1),
                If(self.mac_source.valid,
                   NextValue(z, self.mac_source.payload.z),
                   NextState(next_state),
                )
            )

        fsm.act("WAIT-SINK-VALID",
            # Wait for incoming sample
            self.sink.ready.eq(1),
            # Latch it and start processing
            If(self.sink.valid,
               NextValue(self.x, self.sink.payload.sample),
               NextState("MAC-RESONANCE"),
            )
        )
        fsm_mac("MAC-RESONANCE", "SATURATION",
                self.x - self.y, self.resonance, self.x, self.rezz)
        fsm.act("SATURATION",
            If(self.rezz > SAT_HI,
                NextValue(self.sat, SAT_HI),
            ).Elif(self.rezz < SAT_LO,
                NextValue(self.sat, SAT_LO),
            ).Else(
                NextValue(self.sat, self.rezz),
            ),
            NextState("MAC-LADDER0"),
        )
        fsm_mac("MAC-LADDER0", "MAC-LADDER1",
                self.sat - self.a1, self.g, self.a1, self.a1)
        fsm_mac("MAC-LADDER1", "MAC-LADDER2",
                self.a1 - self.a2, self.g, self.a2, self.a2)
        fsm_mac("MAC-LADDER2", "MAC-LADDER3",
                self.a2 - self.a3, self.g, self.a3, self.a3)
        fsm_mac("MAC-LADDER3", "WAIT-SOURCE-READY",
                self.a3 - self.y, self.g, self.y, self.y)
        fsm.act("WAIT-SOURCE-READY",
            self.source.valid.eq(1),
            self.source.payload.sample.eq(self.y),
            If(self.source.ready,
               NextValue(self.source.valid, 0),
               NextState("WAIT-SINK-VALID"),
            )
        )

class DelayLine(Module):
    """
    - Accepts sample_write, each one written to an incrementing
    index in a local circular buffer.
    - Accepts delay_addr, each of which emits a sample_read, which
    is the value of the sample delay_addr elements later than the
    last 'sample_write' to occur up to MAX_DELAY.

    Note: NeoPixel example in led.py has example of wishbone-mapped
    Memory that could be used for wavetable loading?
    """
    def __init__(self, sw=16, dw=32, fbits=16, max_delay=512):
        stype = (sw, True) # signed, sw-wide samples
        dtype = (dw, True) # signed, dw-wide delays

        # TODO: ensure max_delay is POW2!

        # Sink for writes to Delayline
        self.wsink = Endpoint([("sample", stype)])

        # Source/sink for reads
        self.sink  = Endpoint([("delay",  dtype)])
        self.source = Endpoint([("sample", stype)])

        # Create memory and write pointer
        storage = Memory(width=sw, depth=max_delay)
        # Register, incremented on every write.
        wrpointer = Signal(max=max_delay, reset=0)
        # Wire, address into memory of current read.
        rdpointer = Signal(max=max_delay)

        # Create dual-port memory
        wport = storage.get_port(write_capable=True)
        # Async read is necessary for passthrough combinatorial read sink/source
        rport = storage.get_port(async_read=True)
        self.specials += [
            storage,
            wport,
            rport
        ]

        # Connect up read side
        #
        # self.sink: fractional delay, top (dw-fbits) index into the buffer
        #            at current sample + delay samples (wrapped).
        # self.source: sample at this position in the buffer
        self.comb += [
            # Read pointer must be wrapped to max delay
            # Should wrap correctly as long as max delay is POW2
            rdpointer.eq(
                (wrpointer - (self.sink.delay >> fbits))
            ),
            rport.adr.eq(rdpointer),
            self.source.sample.eq(rport.dat_r),
            # Connect as stream CombinatorialActor
            self.source.valid.eq(self.sink.valid),
            self.source.first.eq(self.sink.first),
            self.source.last.eq(self.sink.last),
            self.sink.ready.eq(self.source.ready),
        ]

        # Connect up write side
        #
        # self.wsink: stream => circular buffer
        self.comb += [
            wport.adr.eq(wrpointer),
            wport.we.eq(self.wsink.valid),
            wport.dat_w.eq(self.wsink.sample),
            self.wsink.ready.eq(1),
        ]

        # Increment circular buffer pointer on every write operation.
        self.sync += [
            If(wport.we,
               If(wrpointer != (max_delay - 1),
                   wrpointer.eq(wrpointer + 1)
               ).Else(
                   wrpointer.eq(0)
               )
            )
        ]

class PitchShift(Module):
    def __init__(self, delayln, mac, xfade=64, sw=16, dw=32, fbits=16):
        stype = (sw, True) # signed, sw-wide samples
        dtype = (dw, True) # signed, dw-wide pitch

        # Source for next sample
        self.source = Endpoint([("sample", stype)])

        # Strobe to produce next sample.
        # Produces a new sample on transition LO -> HI
        self.sample_strobe = Signal()

        # Current pitch shift
        self.pitch = Signal(dtype)
        # Size of grains / pitch shift window
        self.window_sz = Signal(16)

        # Delayline, MAC ports (TODO: schedule 2x in sequence)
        self.delayln_sink, self.delayln_source = delayln.get_port()
        self.mac_sink, self.mac_source = mac.get_port()

        # Current position in delay line 0 (+= pitch every sample)
        self.delay0 = Signal(dtype)
        # Current position in delay line 1
        self.delay1 = Signal(dtype)
        # Last samples from delay lines
        self.sample0 = Signal(stype)
        self.sample1 = Signal(stype)
        # Envelope values
        self.env0 = Signal(dtype)
        self.env1 = Signal(dtype)
        # Last samples from delay lines after MAC operation
        self.scaled0 = Signal(dtype)
        self.scaled1 = Signal(dtype)

        bits_in_xfade = int(math.log2(xfade))

        self.comb += [
            self.delay1.eq(self.delay0 + (self.window_sz << fbits))
        ]

        fsm = FSM(reset_state="WAIT-STROBE")
        self.submodules += fsm

        fsm.act("WAIT-STROBE",
            If(self.sample_strobe,
               # TODO: do this at end?
               If(self.delay0 + self.pitch < 0,
                   NextValue(self.delay0, self.delay0 + self.pitch + ((self.window_sz - 1) << fbits)),
               ).Elif(self.delay0 + self.pitch > (self.window_sz << fbits),
                   NextValue(self.delay0, self.delay0 + self.pitch - ((self.window_sz - 1)<< fbits)),
               ).Else(
                   NextValue(self.delay0, self.delay0 + self.pitch),
               ),
               NextState("WAIT-DELAY0"),
            )
        )
        fsm.act("WAIT-DELAY0",
            self.delayln_sink.payload.delay.eq(self.delay0),
            self.delayln_sink.valid.eq(1),
            self.delayln_source.ready.eq(1),
            If(self.delayln_source.valid,
               NextValue(self.sample0, self.delayln_source.payload.sample),
               NextState("WAIT-DELAY1"),
            ),
        )
        fsm.act("WAIT-DELAY1",
            self.delayln_sink.payload.delay.eq(self.delay1),
            self.delayln_sink.valid.eq(1),
            self.delayln_source.ready.eq(1),
            If(self.delayln_source.valid,
               NextValue(self.sample1, self.delayln_source.payload.sample),
               NextState("WAIT-ENV"),
            ),
        )
        fsm.act("WAIT-ENV",
           If((self.delay0 >> fbits) < xfade,
               # to verify: should be 0x0 -> 0xFFFF
               NextValue(self.env0, (self.delay0 >> bits_in_xfade)),
               NextValue(self.env1, float_to_fp(1.0) - (self.delay0 >> bits_in_xfade)),
           ).Else(
               NextValue(self.env0, float_to_fp(1.0)),
               NextValue(self.env1, 0),
           ),
           NextState("WAIT-MAC0"),
        )
        fsm.act("WAIT-MAC0",
            self.mac_sink.payload.a.eq(self.sample0), #TODO: this is 16->32, verify.
            self.mac_sink.payload.b.eq(self.env0),
            self.mac_sink.payload.c.eq(0),
            self.mac_sink.valid.eq(1),
            self.mac_source.ready.eq(1),
            If(self.mac_source.valid,
               NextValue(self.scaled0, self.mac_source.payload.z),
               NextState("WAIT-MAC1"),
            )
        )
        fsm.act("WAIT-MAC1",
            self.mac_sink.payload.a.eq(self.sample1),
            self.mac_sink.payload.b.eq(self.env1),
            self.mac_sink.payload.c.eq(0),
            self.mac_sink.valid.eq(1),
            self.mac_source.ready.eq(1),
            If(self.mac_source.valid,
               NextValue(self.scaled1, self.mac_source.payload.z),
               NextState("WAIT-SOURCE-READY"),
            )
        )
        fsm.act("WAIT-SOURCE-READY",
            self.source.valid.eq(1),
            self.source.payload.sample.eq(self.scaled0 + self.scaled1),
            If(self.source.ready,
               NextValue(self.source.valid, 0),
               NextState("WAIT-STROBE"),
            )
        )

class TestDSP(unittest.TestCase):
    def test_fixmac(self):
        dut = FixMac()
        #print(verilog.convert(dut))
        def generator(dut):
            yield dut.sink.payload.a.eq(float_to_fp(0.5))
            yield dut.sink.payload.b.eq(float_to_fp(3))
            yield dut.sink.payload.c.eq(float_to_fp(10))
            yield
            result = yield dut.source.payload.z
            self.assertEqual(11.5, fp_to_float(result))
        run_simulation(dut, generator(dut), vcd_name="test_fixmac.vcd")

    def test_rrmac(self):
        dut = RRMux(n=2, inner=FixMac())
        sink0, source0 = dut.get_port()
        sink1, source1 = dut.get_port()
        #print(verilog.convert(dut))
        def generator(dut):
            yield sink0.payload.a.eq(float_to_fp(1.25))
            yield sink0.payload.b.eq(float_to_fp(1.5))
            yield sink0.payload.c.eq(float_to_fp(0.0))
            yield sink0.valid.eq(1)
            yield sink1.payload.a.eq(float_to_fp(0.25))
            yield sink1.payload.b.eq(float_to_fp(0.5))
            yield sink1.payload.c.eq(float_to_fp(4.0))
            yield sink1.valid.eq(1)
            for _ in range(10):
                yield
                result0 = yield source0.payload.z
                result0_valid = yield source0.valid
                result1 = yield source1.payload.z
                result1_valid = yield source1.valid
                print(list(map(fp_to_float, [result0, result1])))
                if result0_valid:
                    self.assertEqual(float_to_fp(1.875), result0)
                if result1_valid:
                    self.assertEqual(float_to_fp(4.125), result1)
        run_simulation(dut, generator(dut), vcd_name="test_rrmac.vcd")

    def test_dcblock_single(self):
        print()

        class DcBlockDUT(Module):
            def __init__(self):
                self.submodules.rrmac = RRMux(n=2, inner=FixMac())
                self.submodules.dc0 = DcBlock(mac=self.rrmac)
                self.submodules.dc1 = DcBlock(mac=self.rrmac)

        dut = DcBlockDUT()

        def generator(dut):
            dc = dut.dc0
            samples = [0.0] + [1.0]*10
            yield dc.source.ready.eq(1)
            for sample in samples:
                # Wait for DUT to be ready for a sample
                while not (yield dc.sink.ready):
                    print("wait for dc.sink.ready...")
                    yield
                # Clock in 1 sample
                print("clock in 1 sample")
                yield dc.sink.payload.sample.eq(float_to_fp(sample))
                yield dc.sink.valid.eq(1)
                yield
                # Wait for the output
                yield dc.sink.valid.eq(0)
                while (yield dc.source.valid != 1):
                    print("wait for dc.source.valid...")
                    yield
                sample_out = yield dc.source.payload.sample
                print ("out", hex(sample_out), fp_to_float(sample_out))
        run_simulation(dut, generator(dut), vcd_name="test_dcblock.vcd")

    def test_dcblock_n(self):
        print()
        class DcBlockDUT(Module):
            def __init__(self):
                self.submodules.rrmac = RRMux(n=2, inner=FixMac())
                self.submodules.dc0 = DcBlock(mac=self.rrmac)
                self.submodules.dc1 = DcBlock(mac=self.rrmac)

        dut = DcBlockDUT()

        def generator(dut):
            dcs = [dut.dc0, dut.dc1]
            samples0 = [0.0] + [1.0]*10
            samples1 = [0.0] + [-2.0]*10
            for dc in dcs:
                yield dc.source.ready.eq(1)
            for sample0, sample1 in zip(samples0, samples1):
                # Wait for DUT to be ready for a sample
                while not all((yield [dc.sink.ready for dc in dcs])):
                    print("wait for both dc.sink.ready...")
                    yield
                # Clock in 1 sample
                print("clock in 1 sample")
                for dc, sample in zip(dcs, [sample0, sample1]):
                    yield dc.sink.payload.sample.eq(float_to_fp(sample))
                    yield dc.sink.valid.eq(1)
                yield
                # Wait for the output
                for dc in dcs:
                    yield dc.sink.valid.eq(0)
                n_results = 0
                while n_results != len(dcs):
                    for dc in dcs:
                        if (yield dc.source.valid == 1):
                            sample_out = yield dc.source.payload.sample
                            print ("out", hex(sample_out), fp_to_float(sample_out))
                            n_results += 1
                    print("wait for dc.source.valid...")
                    yield
        run_simulation(dut, generator(dut), vcd_name="test_dcblock.vcd")

    """
    def test_ladder_lpf_single(self):
        print()

        class LadderDUT(Module):
            def __init__(self):
                self.submodules.rrmac = RRMux(n=2, inner=FixMac())
                self.submodules.lpf = LadderLpf(mac=self.rrmac)

        dut = LadderDUT()
        #print(verilog.convert(dut))

        def generator(dut):
            lpf = dut.lpf
            samples = [0.2 * math.sin((n / 20) * 2*math.pi) +
                       0.2 * math.sin((n / 5) * 2*math.pi) for n in range(100)]
            print(samples)
            yield lpf.source.ready.eq(1)
            for g_set in [1, 0.5, 0.25, 0.125]:
                yield lpf.g.eq(float_to_fp(g_set))
                yield lpf.resonance.eq(float_to_fp(0.0))
                for sample in samples:
                    # Wait for DUT to be ready for a sample
                    while not (yield lpf.sink.ready):
                        print("wait for lpf.sink.ready...")
                        yield
                    # Clock in 1 sample
                    print("clock in 1 sample")
                    yield lpf.sink.payload.sample.eq(float_to_fp(sample))
                    yield lpf.sink.valid.eq(1)
                    yield
                    # Wait for the output
                    yield lpf.sink.valid.eq(0)
                    while (yield lpf.source.valid != 1):
                        #print("wait for lpf.source.valid...")
                        yield
                    sample_out = yield lpf.source.payload.sample
                    print ("out", hex(sample_out), fp_to_float(sample_out))
        run_simulation(dut, generator(dut), vcd_name="test_lpf.vcd")
        """

    def test_delayline(self):
        print()

        class DelayLineDUT(Module):
            def __init__(self):
                self.submodules.delayln = DelayLine(max_delay=32)

        dut = DelayLineDUT()

        #print(verilog.convert(dut))

        def generator(dut):
            samples = [0]*2 + [0xDEAD, 0xBEEF] + [0]*16
            delayln = dut.delayln
            yield delayln.sink.valid.eq(1)
            yield delayln.source.ready.eq(1)
            for sample in samples:
                # Write
                yield delayln.wsink.valid.eq(1)
                yield delayln.wsink.payload.sample.eq(sample)
                yield
                print(f". in={hex(sample)}\tout1=---\tout2=---\tout3=---")
                # Stop write
                yield delayln.wsink.valid.eq(0)
                # Delay 1
                yield delayln.sink.payload.delay.eq(float_to_fp(6))
                yield
                sample_out = (yield delayln.source.payload.sample) & 0xFFFF
                print(f". in=---\tout1={hex(sample_out)}\tout2=---\tout3=---")
                # Delay 2
                yield delayln.sink.payload.delay.eq(float_to_fp(2))
                yield
                sample_out = (yield delayln.source.payload.sample) & 0xFFFF
                print(f". in=---\tout1=---\tout2={hex(sample_out)}\tout3=---")
                # Delay 3
                yield delayln.sink.payload.delay.eq(float_to_fp(9))
                yield
                sample_out = (yield delayln.source.payload.sample) & 0xFFFF
                print(f". in=---\tout1=---\tout2=---\tout3={hex(sample_out)}")

        run_simulation(dut, generator(dut), vcd_name="test_delayline.vcd")


    def test_pitch_shift(self):
        print()

        class PitchShiftDUT(Module):
            def __init__(self):
                self.submodules.rrdelayln = RRMux(n=2, inner=DelayLine(max_delay=128))
                self.submodules.rrmac = RRMux(n=2, inner=FixMac())
                self.submodules.shifter = PitchShift(delayln=self.rrdelayln,
                                                     mac=self.rrmac,
                                                     xfade=32)

        dut = PitchShiftDUT()

        #print(verilog.convert(dut))

        def generator(dut):
            # Default no pitch shift == 1
            # 1 == DC
            # 0.75 == much slower
            # 0.25 == a bit slower
            # 0 == passthrough
            # -0.5 == 1.5x fast?
            # -1 == 2x fast
            # -2 == 3x fast
            yield dut.shifter.pitch.eq(float_to_fp(-0.5))
            yield dut.shifter.window_sz.eq(64) # 20 => max delay 40

            samples = [0.2 * math.sin((n / 15) * 2*math.pi) for n in range(400)]

            delayln = dut.rrdelayln.inner
            yield dut.shifter.source.ready.eq(1)
            for sample in samples:
                # Write to shared delayline
                yield delayln.wsink.valid.eq(1)
                yield delayln.wsink.payload.sample.eq(float_to_fp(sample))
                yield
                yield delayln.wsink.valid.eq(0)
                # Strobe the pitch shifter
                yield dut.shifter.sample_strobe.eq(1)
                yield
                yield dut.shifter.sample_strobe.eq(0)
                # Wait until we get an output sample
                while (yield dut.shifter.source.valid != 1):
                    #print("wait for dut.shifter.source.valid...")
                    yield
                sample_out = yield dut.shifter.source.payload.sample
                print ("out", hex(sample_out), fp_to_float(sample_out))

        run_simulation(dut, generator(dut), vcd_name="test_pitch_shift.vcd")
