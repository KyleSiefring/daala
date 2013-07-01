/*Daala video codec
Copyright (c) 2006-2013 Daala project contributors.  All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

- Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

- Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "decint.h"
#include "generic_code.h"
#include "filter.h"
#include "dct.h"
#include "intra.h"
#include "pvq.h"
#include "pvq_code.h"
#include "block_size.h"
#include "block_size_dec.h"
#include "tf.h"

static int od_dec_init(od_dec_ctx *dec, const daala_info *info,
 const daala_setup_info *setup) {
  int ret;
  (void)setup;
  ret = od_state_init(&dec->state, info);
  if (ret < 0) return ret;
  dec->packet_state = OD_PACKET_DATA;
  return 0;
}

static void od_dec_clear(od_dec_ctx *dec) {
  od_state_clear(&dec->state);
}

daala_dec_ctx *daala_decode_alloc(const daala_info *info,
 const daala_setup_info *setup) {
  od_dec_ctx *dec;
  if (info == NULL) return NULL;
  dec = (od_dec_ctx *)_ogg_malloc(sizeof(*dec));
  if (od_dec_init(dec, info, setup) < 0) {
    _ogg_free(dec);
    return NULL;
  }
  return dec;
}

void daala_decode_free(daala_dec_ctx *dec) {
  if (dec != NULL) {
    od_dec_clear(dec);
    _ogg_free(dec);
  }
}

int daala_decode_ctl(daala_dec_ctx *dec, int req, void *buf, size_t buf_sz) {
  (void)dec;
  (void)buf;
  (void)buf_sz;
  switch(req) {
    default: return OD_EIMPL;
  }
}

static void od_decode_mv(daala_dec_ctx *dec, od_mv_grid_pt *mvg, int vx,
 int vy, int level, int mv_res, int width, int height) {
  static const int ex[5] = {628, 1382, 1879, 2119, 2102};
  static const int ey[5] = {230, 525, 807, 1076, 1332};
  int pred[2];
  int ox;
  int oy;
  ox = laplace_decode(&dec->ec, ex[level] >> mv_res, width << (4 - level));
  oy = laplace_decode(&dec->ec, ey[level] >> mv_res, height << (4 - level));
  /*Deinterleave positive and negative values.*/
  ox = (ox >> 1) ^ -(ox & 1);
  oy = (oy >> 1) ^ -(oy & 1);
  od_state_get_predictor(&dec->state, pred, vx, vy, level, mv_res);
  mvg->mv[0] = (pred[0] + ox) << mv_res;
  mvg->mv[1] = (pred[1] + oy) << mv_res;
}

struct od_mb_dec_ctx {
  od_coeff tfbuf[16*16*4];
  generic_encoder model_dc[OD_NPLANES_MAX];
  generic_encoder model_g[OD_NPLANES_MAX];
  generic_encoder model_ym[OD_NPLANES_MAX];
  od_adapt_ctx adapt;
  signed char *modes;
  od_coeff *c;
  od_coeff *d;
  od_coeff *md;
  od_coeff *mc;
  od_coeff *l;
  int run_pvq[OD_NPLANES_MAX];
  int ex_dc[OD_NPLANES_MAX];
  int ex_g[OD_NPLANES_MAX];
  int is_keyframe;
  int nk;
  int k_total;
  int sum_ex_total_q8;
  int ncount;
  int count_total_q8;
  int count_ex_total_q8;
  ogg_uint16_t mode_p0[OD_INTRA_NMODES];
};
typedef struct od_mb_dec_ctx od_mb_dec_ctx;

void od_single_band_decode(daala_dec_ctx *dec, od_mb_dec_ctx *ctx, int ln,
 int pli, int bx, int by, int has_ur) {
  int n;
  int n2;
  int xdec;
  int ydec;
  int w;
  int frame_width;
  signed char *modes;
  od_coeff *c;
  od_coeff *d;
  od_coeff *md;
  od_coeff *mc;
  od_coeff *l;
  int x;
  int y;
  od_coeff pred[16*16];
  od_coeff predt[16*16];
  ogg_int16_t pvq_scale[16*16];
  int sgn;
  int qg;
  int zzi;
  int vk;
  int run_pvq;
  unsigned char const *zig;
  OD_ASSERT(ln >= 0 && ln <= 2);
  run_pvq = ctx->run_pvq[pli];
  n = 1 << (ln + 2);
  n2 = n*n;
  bx <<= ln;
  by <<= ln;
  zig = OD_DCT_ZIGS[ln];
  xdec = dec->state.io_imgs[OD_FRAME_INPUT].planes[pli].xdec;
  ydec = dec->state.io_imgs[OD_FRAME_INPUT].planes[pli].ydec;
  frame_width = dec->state.frame_width;
  w = frame_width >> xdec;
  modes = ctx->modes;
  c = ctx->c;
  d = ctx->d;
  md = ctx->md;
  mc = ctx->mc;
  l = ctx->l;
  vk = 0;
  /*Apply forward transform to MC predictor.*/
  if (!ctx->is_keyframe) {
    (*OD_FDCT_2D[ln])(md + (by << 2)*w + (bx << 2), w,
     mc + (by << 2)*w + (bx << 2), w);
  }
  for (zzi = 0; zzi < n2; zzi++) pvq_scale[zzi] = 0;
  if (ctx->is_keyframe) {
    /*Chroma from luma temporarily disabled until it handles switching.*/
    if (bx > 0 && by > 0 && pli == 0) {
      if (pli == 0) {
        ogg_uint16_t mode_cdf[OD_INTRA_NMODES];
        int m_l;
        int m_ul;
        int m_u;
        int mode;
        od_coeff *coeffs[4];
        int strides[4];
        /*Calculate the intra-prediction.*/
        coeffs[0] = &ctx->tfbuf[0];
        coeffs[1] = &ctx->tfbuf[n2];
        coeffs[2] = &ctx->tfbuf[n2*2];
        coeffs[3] = &ctx->tfbuf[n2*3];
        strides[0] = n;
        strides[1] = n;
        strides[2] = n;
        strides[3] = n;
        od_convert_intra_coeffs(coeffs, strides, d,
         w, bx, by, dec->state.bsize, dec->state.bstride, has_ur);
        m_l = modes[by*(w >> 2) + bx - 1];
        m_ul = modes[(by - 1)*(w >> 2) + bx - 1];
        m_u = modes[(by - 1)*(w >> 2) + bx];
        od_intra_pred_cdf(mode_cdf, OD_INTRA_PRED_PROB_4x4[pli],
         ctx->mode_p0, OD_INTRA_NMODES, m_l, m_ul, m_u);
        mode = od_ec_decode_cdf_unscaled(&dec->ec, mode_cdf,
         OD_INTRA_NMODES);
        (*OD_INTRA_GET[ln])(pred, coeffs, strides, mode);
        for (y = 0; y < (1 << ln); y++) {
          for (x = 0; x < (1 << ln); x++) {
            modes[(by + y)*(w >> 2) + bx + x] = mode;
          }
        }
        od_intra_pred_update(ctx->mode_p0, OD_INTRA_NMODES, mode,
         m_l, m_ul, m_u);
      }
      else{
        int chroma_weights_q8[3];
        int mode;
        mode = modes[(by << ydec)*(frame_width >> 2) + (bx << xdec)];
        chroma_weights_q8[0] = OD_INTRA_CHROMA_WEIGHTS_Q6[mode][0];
        chroma_weights_q8[1] = OD_INTRA_CHROMA_WEIGHTS_Q6[mode][1];
        chroma_weights_q8[2] = OD_INTRA_CHROMA_WEIGHTS_Q6[mode][2];
        mode = modes[(by << ydec)*(frame_width >> 2)
         + ((bx << xdec) + xdec)];
        chroma_weights_q8[0] +=
         OD_INTRA_CHROMA_WEIGHTS_Q6[mode][0];
        chroma_weights_q8[1] +=
         OD_INTRA_CHROMA_WEIGHTS_Q6[mode][1];
        chroma_weights_q8[2] +=
         OD_INTRA_CHROMA_WEIGHTS_Q6[mode][2];
        mode = modes[((by << ydec) + ydec)*(frame_width >> 2)
         + (bx << xdec)];
        chroma_weights_q8[0] +=
         OD_INTRA_CHROMA_WEIGHTS_Q6[mode][0];
        chroma_weights_q8[1] +=
         OD_INTRA_CHROMA_WEIGHTS_Q6[mode][1];
        chroma_weights_q8[2] +=
         OD_INTRA_CHROMA_WEIGHTS_Q6[mode][2];
        mode = modes[((by << ydec) + ydec)*(frame_width >> 2)
         + ((bx << xdec) + xdec)];
        chroma_weights_q8[0] +=
         OD_INTRA_CHROMA_WEIGHTS_Q6[mode][0];
        chroma_weights_q8[1] +=
         OD_INTRA_CHROMA_WEIGHTS_Q6[mode][1];
        chroma_weights_q8[2] +=
         OD_INTRA_CHROMA_WEIGHTS_Q6[mode][2];
        od_chroma_pred4x4(pred, d + (by << 2)*w + (bx << 2),
         l + (by << 2)*w + (bx << 2), w, chroma_weights_q8);
      }
    }
    else{
      int nsize;
      for (zzi = 0; zzi < n2; zzi++) pred[zzi] = 0;
      nsize = ln;
      /*444/420 only right now.*/
      OD_ASSERT(xdec==ydec);
      if (bx > 0) {
        int noff;
        nsize = OD_BLOCK_SIZE4x4(dec->state.bsize, dec->state.bstride,
         (bx - 1) << xdec, by << ydec);
        nsize = OD_MAXI(nsize - xdec, 0);
        noff = 1 << nsize;
        /*Because of the quad-tree structure we can always find our neighbors
           starting offset by rounding to a multiple of his size.*/
        OD_ASSERT(!(bx & (noff - 1)));
        pred[0] = d[((by & ~(noff - 1)) << 2)*w + ((bx - noff) << 2)];
      }
      else if (by > 0) {
        int noff;
        nsize = OD_BLOCK_SIZE4x4(dec->state.bsize, dec->state.bstride,
         bx << xdec, (by - 1) << ydec);
        nsize = OD_MAXI(nsize - xdec, 0);
        noff = 1 << nsize;
        OD_ASSERT(!(by & (noff - 1)));
        pred[0] = d[((by - noff) << 2)*w + ((bx & ~(noff - 1)) << 2)];
      }
      /*Rescale DC for correct transform size.*/
      if (nsize > ln) pred[0] >>= (nsize - ln);
      else if (nsize < ln) pred[0] <<= (ln - nsize);
      if (pli == 0) {
        for (y = 0; y < (1 << ln); y++) {
          for (x = 0; x < (1 << ln); x++) {
            modes[(by + y)*(w >> 2) + bx + x] = 0;
          }
        }
      }
    }
  }
  else {
    int ci;
    ci = 0;
    for (y = 0; y < n; y++) {
      for (x = 0; x < n; x++) {
        pred[ci++] = md[(y + (by << 2))*w + (x + (bx << 2))];
      }
    }
  }
  /*Zig-zag*/
  for (y = 0; y < n; y++) {
    for (x = 0; x < n; x++) {
      predt[zig[y*n + x]] = pred[y*n + x];
    }
  }
  if (!run_pvq) {
    int scale;
    scale = OD_MAXI(dec->scale[pli], 1);
    pred[0] = generic_decode(&dec->ec, ctx->model_dc + pli,
     ctx->ex_dc + pli, 0);
    if (pred[0]) pred[0] *= od_ec_dec_bits(&dec->ec, 1) ? -1 : 1;
    pred[0] = pred[0]*scale + predt[0];
    vk = generic_decode(&dec->ec, ctx->model_g + pli, ctx->ex_g + pli, 0);
    pvq_decoder(&dec->ec, pred + 1, n2 - 1, vk, &ctx->adapt);
    for (zzi = 1; zzi < n2; zzi++) pred[zzi] = pred[zzi]*scale + predt[zzi];
  }
  if (run_pvq) {
    int scale;
    scale = OD_MAXI((dec->scale[pli]*OD_TRANS_QUANT_ADJ[ln]
     + (1 << 14)) >> 15, 1);
    sgn = 0;
    pred[0] = generic_decode(&dec->ec, ctx->model_dc + pli,
     ctx->ex_dc + pli, 0);
    if (pred[0]) sgn = od_ec_dec_bits(&dec->ec, 1);
    pred[0] = (int)(pow(pred[0], 4.0/3)*scale + 0.5);
    pred[0] *= sgn ? -1 : 1;
    pred[0] += predt[0];
    qg = generic_decode(&dec->ec, ctx->model_g + pli, ctx->ex_g + pli, 0);
    if (qg) qg *= od_ec_dec_bits(&dec->ec, 1) ? -1 : 1;
    vk = pvq_unquant_k(&predt[1], n2 - 1, qg, scale, 4 - ln, ctx->is_keyframe);
    pred[1] = 0;
    if (vk != 0) {
      int ex_ym;
      ex_ym = (65536/2)*vk;
      pred[1] = vk - generic_decode(&dec->ec, ctx->model_ym + pli, &ex_ym, 0);
    }
    pvq_decoder(&dec->ec, pred + 2, n2 - 2, vk - abs(pred[1]), &ctx->adapt);
    dequant_pvq(pred + 1, predt + 1, pvq_scale, n2 - 1, scale, qg, 4 - ln,
     ctx->is_keyframe);
  }
  if (ctx->adapt.curr[OD_ADAPT_K_Q8] >= 0) {
    ctx->nk++;
    ctx->k_total += ctx->adapt.curr[OD_ADAPT_K_Q8];
    ctx->sum_ex_total_q8 += ctx->adapt.curr[OD_ADAPT_SUM_EX_Q8];
  }
  if (ctx->adapt.curr[OD_ADAPT_COUNT_Q8] >= 0) {
    ctx->ncount++;
    ctx->count_total_q8 += ctx->adapt.curr[OD_ADAPT_COUNT_Q8];
    ctx->count_ex_total_q8 += ctx->adapt.curr[OD_ADAPT_COUNT_EX_Q8];
  }
  /*De-zigzag*/
  for (y = 0; y < n; y++) {
    for (x = 0; x < n; x++) {
      d[((by << 2) + y)*w + (bx << 2) + x] = pred[zig[y*n + x]];
    }
  }
  /*Apply the inverse transform.*/
  (*OD_IDCT_2D[ln])(c + (by << 2)*w + (bx << 2), w,
   d + (by << 2)*w + (bx << 2), w);
}

static void od_32x32_decode(daala_dec_ctx *dec, od_mb_dec_ctx *ctx, int ln,
 int pli, int bx, int by, int has_ur) {
  bx <<= 1;
  by <<= 1;
  od_single_band_decode(dec, ctx, ln - 1, pli, bx + 0, by + 0, 1);
  od_single_band_decode(dec, ctx, ln - 1, pli, bx + 1, by + 0, has_ur);
  od_single_band_decode(dec, ctx, ln - 1, pli, bx + 0, by + 1, 1);
  od_single_band_decode(dec, ctx, ln - 1, pli, bx + 1, by + 1, 0);
}

typedef void (*od_dec_func)(daala_dec_ctx *dec, od_mb_dec_ctx *ctx, int ln,
 int pli, int bx, int by, int has_ur);

const od_dec_func OD_DECODE_BLOCK[OD_NBSIZES + 2]={
  od_single_band_decode,
  od_single_band_decode,
  od_single_band_decode,
  od_32x32_decode
};

static void od_decode_block(daala_dec_ctx *dec, od_mb_dec_ctx *ctx, int pli,
 int bx, int by, int l, int xdec, int ydec, int has_ur) {
  int d;
  /*This code assumes 4:4:4 or 4:2:0 input.*/
  OD_ASSERT(xdec == ydec);
  d = OD_MAXI(OD_BLOCK_SIZE4x4(dec->state.bsize,
   dec->state.bstride, bx << l, by << l), xdec);
  OD_ASSERT(d <= l);
  if (d == l) {
    (*OD_DECODE_BLOCK[d - xdec])(dec, ctx, d - xdec, pli, bx, by, has_ur);
  }
  else {
    l--;
    bx <<= 1;
    by <<= 1;
    od_decode_block(dec, ctx, pli, bx + 0, by + 0, l, xdec, ydec, 1);
    od_decode_block(dec, ctx, pli, bx + 1, by + 0, l, xdec, ydec, has_ur);
    od_decode_block(dec, ctx, pli, bx + 0, by + 1, l, xdec, ydec, 1);
    od_decode_block(dec, ctx, pli, bx + 1, by + 1, l, xdec, ydec, 0);
  }
}

int daala_decode_packet_in(daala_dec_ctx *dec, od_img *img,
 const ogg_packet *op) {
  int nplanes;
  int pli;
  int frame_width;
  int frame_height;
  int nvsb;
  int nhsb;
  int refi;
  int i;
  int j;
  od_mb_dec_ctx mbctx;
  if (dec == NULL || img == NULL || op == NULL) return OD_EFAULT;
  if (dec->packet_state != OD_PACKET_DATA) return OD_EINVAL;
  if (op->e_o_s) dec->packet_state = OD_PACKET_DONE;
  od_ec_dec_init(&dec->ec, op->packet, op->bytes);
  /*Read the packet type bit.*/
  if (od_ec_decode_bool_q15(&dec->ec, 16384)) return OD_EBADPACKET;
  mbctx.is_keyframe = od_ec_decode_bool_q15(&dec->ec, 16384);
  /*Update the buffer state.*/
  if (dec->state.ref_imgi[OD_FRAME_SELF] >= 0) {
    dec->state.ref_imgi[OD_FRAME_PREV] =
     dec->state.ref_imgi[OD_FRAME_SELF];
    /*TODO: Update golden frame.*/
    if (dec->state.ref_imgi[OD_FRAME_GOLD] < 0) {
      dec->state.ref_imgi[OD_FRAME_GOLD] =
       dec->state.ref_imgi[OD_FRAME_SELF];
      /*TODO: Mark keyframe timebase.*/
    }
  }
  /*Select a free buffer to use for this reference frame.*/
  for (refi = 0; refi == dec->state.ref_imgi[OD_FRAME_GOLD]
   || refi == dec->state.ref_imgi[OD_FRAME_PREV]
   || refi == dec->state.ref_imgi[OD_FRAME_NEXT]; refi++);
  dec->state.ref_imgi[OD_FRAME_SELF] = refi;
  nhsb = dec->state.nhsb;
  nvsb = dec->state.nvsb;
  for(i = -4; i < (nhsb+1)*4; i++) {
    for(j = -4; j < 0; j++) {
      dec->state.bsize[(j*dec->state.bstride) + i] = 3;
    }
    for(j = nvsb*4; j < (nvsb+1)*4; j++) {
      dec->state.bsize[(j*dec->state.bstride) + i] = 3;
    }
  }
  for(j = -4; j < (nvsb+1)*4; j++) {
    for(i = -4; i < 0; i++) {
      dec->state.bsize[(j*dec->state.bstride) + i] = 3;
    }
    for(i = nhsb*4; i < (nhsb+1)*4; i++) {
      dec->state.bsize[(j*dec->state.bstride) + i] = 3;
    }
  }
  for(i = 0; i < nvsb; i++) {
    for(j = 0; j < nhsb; j++) {
      od_block_size_decode(&dec->ec,
       &dec->state.bsize[4*dec->state.bstride*i + 4*j], dec->state.bstride);
    }
  }
  if(!mbctx.is_keyframe){
    /* Input the motion vectors. */
    int nhmvbs;
    int nvmvbs;
    int vx;
    int vy;
    od_img *img;
    int width;
    int height;
    int mv_res;
    od_mv_grid_pt *mvp;
    od_mv_grid_pt **grid;
    OD_ASSERT(dec->state.ref_imgi[OD_FRAME_PREV] >= 0);
    od_state_mvs_clear(&dec->state);
    nhmvbs = (dec->state.nhmbs + 1) << 2;
    nvmvbs = (dec->state.nvmbs + 1) << 2;
    img = dec->state.io_imgs + OD_FRAME_REC;
    mv_res = dec->state.mv_res = od_ec_dec_uint(&dec->ec, 3);
    width = (img->width + 32) << (3 - mv_res);
    height = (img->height + 32) << (3 - mv_res);
    grid = dec->state.mv_grid;
    /*Level 0.*/
    for (vy = 0; vy <= nvmvbs; vy += 4) {
      for (vx = 0; vx <= nhmvbs; vx += 4) {
        mvp = &grid[vy][vx];
        mvp->valid = 1;
        od_decode_mv(dec, mvp, vx, vy, 0, mv_res, width, height);
      }
    }
    /*Level 1.*/
    for (vy = 2; vy <= nvmvbs; vy += 4) {
      for (vx = 2; vx <= nhmvbs; vx += 4) {
        int p_invalid;
        p_invalid = od_mv_level1_prob(grid,vx,vy);
        mvp = &grid[vy][vx];
        mvp->valid = od_ec_decode_bool_q15(&dec->ec, p_invalid);
        if (mvp->valid) {
          od_decode_mv(dec, mvp, vx, vy, 1, mv_res, width, height);
        }
      }
    }
    /*Level 2.*/
    for (vy = 0; vy <= nvmvbs; vy += 2) {
      for (vx = 2*((vy & 3) == 0); vx <= nhmvbs; vx += 4) {
        mvp = &grid[vy][vx];
        if ((vy-2 < 0 || grid[vy-2][vx].valid)
         && (vx-2 < 0 || grid[vy][vx-2].valid)
         && (vy+2 > nvmvbs || grid[vy+2][vx].valid)
         && (vx+2 > nhmvbs || grid[vy][vx+2].valid)) {
          mvp->valid = od_ec_decode_bool_q15(&dec->ec, 13684);
          if (mvp->valid) {
            od_decode_mv(dec, mvp, vx, vy, 2, mv_res, width, height);
          }
        }
      }
    }
    /*Level 3.*/
    for (vy = 1; vy <= nvmvbs; vy += 2) {
      for (vx = 1; vx <= nhmvbs; vx += 2) {
        mvp = &grid[vy][vx];
        if (grid[vy-1][vx-1].valid && grid[vy-1][vx+1].valid
         && grid[vy+1][vx+1].valid && grid[vy+1][vx-1].valid) {
          mvp->valid = od_ec_decode_bool_q15(&dec->ec, 16384);
          if (mvp->valid) {
            od_decode_mv(dec, mvp, vx, vy, 3, mv_res, width, height);
          }
        }
      }
    }
    /*Level 4.*/
    for (vy = 2; vy <= nvmvbs - 2; vy += 1) {
      for (vx = 3 - (vy & 1); vx <= nhmvbs - 2; vx += 2) {
        mvp = &grid[vy][vx];
        if (grid[vy-1][vx].valid && grid[vy][vx-1].valid
         && grid[vy+1][vx].valid && grid[vy][vx+1].valid) {
          mvp->valid = od_ec_decode_bool_q15(&dec->ec, 16384);
          if (mvp->valid) {
            od_decode_mv(dec, mvp, vx, vy, 4, mv_res, width, height);
          }
        }
      }
    }
    od_state_mc_predict(&dec->state, OD_FRAME_PREV);
  }
  frame_width = dec->state.frame_width;
  frame_height = dec->state.frame_height;
  {
    od_coeff *ctmp[OD_NPLANES_MAX];
    od_coeff *dtmp[OD_NPLANES_MAX];
    od_coeff *mctmp[OD_NPLANES_MAX];
    od_coeff *mdtmp[OD_NPLANES_MAX];
    od_coeff *ltmp[OD_NPLANES_MAX];
    od_coeff *lbuf[OD_NPLANES_MAX];
    int xdec;
    int ydec;
    int sby;
    int sbx;
    int mi;
    int h;
    int w;
    int y;
    int x;
    /*Initialize the data needed for each plane.*/
    mbctx.modes = _ogg_calloc((frame_width >> 2)*(frame_height >> 2),
     sizeof(*mbctx.modes));
    for (mi = 0; mi < OD_INTRA_NMODES; mi++) {
      mbctx.mode_p0[mi] = 32768/OD_INTRA_NMODES;
    }
    nplanes = dec->state.info.nplanes;
    /*Apply the prefilter to the motion-compensated reference.*/
    if (!mbctx.is_keyframe) {
      for (pli = 0; pli < nplanes; pli++) {
        xdec = dec->state.io_imgs[OD_FRAME_REC].planes[pli].xdec;
        ydec = dec->state.io_imgs[OD_FRAME_REC].planes[pli].ydec;
        w = frame_width >> xdec;
        h = frame_height >> ydec;
        mctmp[pli] = _ogg_calloc(w*h, sizeof(*mctmp[pli]));
        mdtmp[pli] = _ogg_calloc(w*h, sizeof(*mdtmp[pli]));
        /*Collect the image data needed for this plane.*/
        {
          unsigned char *mdata;
          int ystride;
          mdata = dec->state.io_imgs[OD_FRAME_REC].planes[pli].data;
          ystride = dec->state.io_imgs[OD_FRAME_REC].planes[pli].ystride;
          for (y=0;y<h;y++) {
            for (x=0;x<w;x++) mctmp[pli][y*w+x]=mdata[ystride*y+x]-128;
          }
        }
        /*Apply the prefilter across the entire image.*/
        for (sby = 0; sby < nvsb; sby++) {
          for (sbx = 0; sbx < nhsb; sbx++) {
            od_apply_prefilter(mctmp[pli], w, sbx, sby, 3, dec->state.bsize,
             dec->state.bstride, xdec, ydec, (sbx > 0 ? OD_LEFT_EDGE : 0) |
             (sby < nvsb - 1 ? OD_BOTTOM_EDGE : 0));
          }
        }
      }
    }
    for (pli = 0; pli < nplanes; pli++) {
      generic_model_init(mbctx.model_dc + pli);
      generic_model_init(mbctx.model_g + pli);
      generic_model_init(mbctx.model_ym + pli);
      mbctx.ex_dc[pli] = pli > 0 ? 8 : 32768;
      mbctx.ex_g[pli] = 8;
      xdec = dec->state.io_imgs[OD_FRAME_INPUT].planes[pli].xdec;
      ydec = dec->state.io_imgs[OD_FRAME_INPUT].planes[pli].ydec;
      w = frame_width >> xdec;
      h = frame_height >> ydec;
      dec->scale[pli] = od_ec_dec_uint(&dec->ec, 512);
      mbctx.run_pvq[pli] = dec->scale[pli] &&
       od_ec_decode_bool_q15(&dec->ec, 16384);
      ctmp[pli] = _ogg_calloc(w*h, sizeof(*ctmp[pli]));
      dtmp[pli] = _ogg_calloc(w*h, sizeof(*dtmp[pli]));
      /*We predict chroma planes from the luma plane.
        Since chroma can be subsampled, we cache subsampled versions of the
         luma plane in the frequency domain.
        We can share buffers with the same subsampling.*/
      if (pli > 0) {
        int plj;
        if (xdec || ydec) {
          for (plj = 1; plj < pli; plj++) {
            if (xdec == dec->state.io_imgs[OD_FRAME_INPUT].planes[plj].xdec
             && ydec == dec->state.io_imgs[OD_FRAME_INPUT].planes[plj].ydec) {
              ltmp[pli] = NULL;
              lbuf[pli] = ltmp[plj];
            }
          }
          if (plj >= pli) {
            lbuf[pli] = ltmp[pli] = _ogg_calloc(w*h, sizeof(*ltmp));
          }
        }
        else{
          ltmp[pli] = NULL;
          lbuf[pli] = ctmp[pli];
        }
      }
      else lbuf[pli] = ltmp[pli] = NULL;
      od_adapt_row_init(&dec->state.adapt_row[pli]);
    }
    for (sby = 0; sby < nvsb; sby++) {
      od_adapt_ctx adapt_hmean[OD_NPLANES_MAX];
      for (pli = 0; pli < nplanes; pli++) {
        od_adapt_hmean_init(&adapt_hmean[pli]);
      }
      for (sbx = 0; sbx < nhsb; sbx++) {
        for (pli = 0; pli < nplanes; pli++) {
          od_adapt_row_ctx *adapt_row;
          int by;
          int bx;
          mbctx.c = ctmp[pli];
          mbctx.d = dtmp[pli];
          mbctx.mc = mctmp[pli];
          mbctx.md = mdtmp[pli];
          mbctx.l = lbuf[pli];
          xdec = dec->state.io_imgs[OD_FRAME_INPUT].planes[pli].xdec;
          ydec = dec->state.io_imgs[OD_FRAME_INPUT].planes[pli].ydec;
          w = frame_width >> xdec;
          h = frame_height >> ydec;
          /*Construct the luma predictors for chroma planes.*/
          if (ltmp[pli] != NULL) {
            OD_ASSERT(pli > 0);
            OD_ASSERT(mbctx.l == ltmp[pli]);
            for (by = sby << (3 - ydec); by < (sby + 1) << (3 - ydec); by++) {
              for (bx = sbx << (3 - xdec); bx < (sbx + 1) << (3 - xdec);
               bx++) {
                od_resample_luma_coeffs(mbctx.l + (by << 2)*w + (bx<<2), w,
                 dtmp[0] + (by << (2 + ydec))*frame_width + (bx<<(2 + xdec)),
                 frame_width, xdec, ydec, 4);
              }
            }
          }
          mbctx.nk = mbctx.k_total = mbctx.sum_ex_total_q8 = 0;
          mbctx.ncount = mbctx.count_total_q8 = mbctx.count_ex_total_q8 = 0;
          adapt_row = &dec->state.adapt_row[pli];
          od_adapt_update_stats(adapt_row, sbx, &adapt_hmean[pli],
           &mbctx.adapt);
          od_decode_block(dec, &mbctx, pli, sbx, sby, 3, xdec, ydec,
           sby > 0 && sbx < nhsb - 1);
          if (mbctx.nk > 0) {
            mbctx.adapt.curr[OD_ADAPT_K_Q8] =
             OD_DIVU_SMALL(mbctx.k_total << 8, mbctx.nk);
            mbctx.adapt.curr[OD_ADAPT_SUM_EX_Q8] =
             OD_DIVU_SMALL(mbctx.sum_ex_total_q8, mbctx.nk);
          } else {
            mbctx.adapt.curr[OD_ADAPT_K_Q8] = OD_ADAPT_NO_VALUE;
            mbctx.adapt.curr[OD_ADAPT_SUM_EX_Q8] = OD_ADAPT_NO_VALUE;
          }
          if (mbctx.ncount > 0)
          {
            mbctx.adapt.curr[OD_ADAPT_COUNT_Q8] =
             OD_DIVU_SMALL(mbctx.count_total_q8, mbctx.ncount);
            mbctx.adapt.curr[OD_ADAPT_COUNT_EX_Q8] =
             OD_DIVU_SMALL(mbctx.count_ex_total_q8, mbctx.ncount);
          } else {
            mbctx.adapt.curr[OD_ADAPT_COUNT_Q8] = OD_ADAPT_NO_VALUE;
            mbctx.adapt.curr[OD_ADAPT_COUNT_EX_Q8] = OD_ADAPT_NO_VALUE;
          }
          od_adapt_sb(adapt_row, sbx, &adapt_hmean[pli], &mbctx.adapt);
        }
      }
      for (pli = 0; pli < nplanes; pli++) {
        od_adapt_row(&dec->state.adapt_row[pli], &adapt_hmean[pli]);
      }
    }
    for (pli = 0; pli < nplanes; pli++) {
      xdec = dec->state.io_imgs[OD_FRAME_INPUT].planes[pli].xdec;
      ydec = dec->state.io_imgs[OD_FRAME_INPUT].planes[pli].ydec;
      w = frame_width >> xdec;
      h = frame_height >> ydec;
      /*Apply the postfilter across the entire image.*/
      for (sby = 0; sby < nvsb; sby++) {
        for (sbx = 0; sbx < nhsb; sbx++) {
          od_apply_postfilter(ctmp[pli], w, sbx, sby, 3, dec->state.bsize,
           dec->state.bstride, xdec, ydec, (sby > 0 ? OD_TOP_EDGE : 0) |
           (sbx < nhsb - 1 ? OD_RIGHT_EDGE : 0));
        }
      }
      {
        unsigned char *data;
        int ystride;
        data = dec->state.io_imgs[OD_FRAME_REC].planes[pli].data;
        ystride = dec->state.io_imgs[OD_FRAME_REC].planes[pli].ystride;
        for (y=0;y<h;y++) {
          for (x=0;x<w;x++) {
            data[ystride*y+x]=OD_CLAMP255(ctmp[pli][y*w+x]+128);
          }
        }
      }
    }
    for (pli = nplanes; pli-- > 0;) {
      _ogg_free(ltmp[pli]);
      _ogg_free(dtmp[pli]);
      _ogg_free(ctmp[pli]);
      if (!mbctx.is_keyframe) {
        _ogg_free(mdtmp[pli]);
        _ogg_free(mctmp[pli]);
      }
    }
    _ogg_free(mbctx.modes);
  }
#if defined(OD_DUMP_IMAGES)
  /*Dump YUV*/
  od_state_dump_yuv(&dec->state, dec->state.io_imgs + OD_FRAME_REC, "decout");
#endif
  od_state_upsample8(&dec->state,
   dec->state.ref_imgs + dec->state.ref_imgi[OD_FRAME_SELF],
   dec->state.io_imgs + OD_FRAME_REC);
  /*Return decoded frame.*/
  *img = dec->state.io_imgs[OD_FRAME_REC];
  img->width = dec->state.info.pic_width;
  img->height = dec->state.info.pic_height;
  dec->state.cur_time++;
  return 0;
}
