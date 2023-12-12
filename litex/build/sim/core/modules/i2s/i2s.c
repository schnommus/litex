#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "error.h"
#include <unistd.h>
#include <math.h>

#include "modules.h"

struct session_s {
    char *sdin1;
    char *sdout1;
    char *lrck;
    char *bick;
    char *mclk;
    uint8_t bit_count;
    uint32_t samples;

    int16_t cur_sample[4];
};

struct event_base *base;
static int litex_sim_module_pads_get(struct pad_s *pads, char *name, void **signal)
{
  int ret = RC_OK;
  void *sig = NULL;
  int i;
  if(!pads || !name || !signal) {
    ret=RC_INVARG;
    goto out;
  }
  i = 0;
  while(pads[i].name) {
    if(!strcmp(pads[i].name, name)) {
      sig = (void*)pads[i].signal;
      break;
    }
    i++;
  }
out:
  *signal = sig;
  return ret;
}

static int i2s_start()
{
  printf("[i2s] loaded\n");
  return RC_OK;
}

static int i2s_new(void **sess, char *args)
{
  int ret = RC_OK;
  struct session_s *s=NULL;
  if(!sess) {
    ret = RC_INVARG;
    goto out;
  }
  s=(struct session_s*)malloc(sizeof(struct session_s));
  if(!s) {
    ret=RC_NOENMEM;
    goto out;
  }
  memset(s, 0, sizeof(struct session_s));
out:
  *sess=(void*)s;
  return ret;
}

static int i2s_add_pads(void *sess, struct pad_list_s *plist)
{
  int ret = RC_OK;
  struct session_s *s = (struct session_s*) sess;
  struct pad_s *pads;
  if(!sess || !plist) {
    ret = RC_INVARG;
    goto out;
  }
  pads = plist->pads;
  if(!strcmp(plist->name, "eurorack_pmod_clk0")) {
    litex_sim_module_pads_get(pads, "lrck", (void**)&s->lrck);
    litex_sim_module_pads_get(pads, "bick", (void**)&s->bick);
    litex_sim_module_pads_get(pads, "mclk", (void**)&s->mclk);
  }
  if(!strcmp(plist->name, "eurorack_pmod0")) {
    litex_sim_module_pads_get(pads, "sdin1", (void**)&s->sdin1);
    litex_sim_module_pads_get(pads, "sdout1", (void**)&s->sdout1);
  }
out:
  return ret;
}

static int i2s_tick(void *sess, uint64_t time_ps) {
  static clk_edge_state_t edge_bick;
  static clk_edge_state_t edge_lrck;

  struct session_s *s = (struct session_s*)sess;

  if(clk_neg_edge(&edge_lrck, *s->lrck)) {
      s->bit_count = 0;
      ++s->samples;
      s->cur_sample[0] = (int16_t)(20000.0*sin(2.0f*M_PI*(float)s->samples / 128.0f));
      s->cur_sample[1] = 0xBEEF;
      s->cur_sample[2] = 0xFEED;
      s->cur_sample[3] = 0xA5D5;
  }

  if(!clk_neg_edge(&edge_bick, *s->bick)) {
    return RC_OK;
  }

  uint8_t bit = s->bit_count % 32;
  uint8_t channel = s->bit_count >> 5;

  if (bit < 16) {
      *s->sdout1 = (s->cur_sample[channel] >> (15-bit)) & 1;
  } else {
      *s->sdout1 = 0;
  }

  ++s->bit_count;

  return RC_OK;
}

static struct ext_module_s ext_mod = {
  "i2s",
  i2s_start,
  i2s_new,
  i2s_add_pads,
  NULL,
  i2s_tick
};

int litex_sim_ext_module_init(int (*register_module) (struct ext_module_s *))
{
  int ret = RC_OK;
  ret = register_module(&ext_mod);
  return ret;
}
