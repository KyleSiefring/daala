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

#if !defined(_encint_H)
# define _encint_H (1)

typedef struct daala_enc_ctx od_enc_ctx;
typedef struct od_params_ctx od_params_ctx;
typedef struct od_mv_est_ctx od_mv_est_ctx;
typedef struct od_enc_opt_vtbl od_enc_opt_vtbl;
typedef struct od_rollback_buffer od_rollback_buffer;

# include "../include/daala/daaladec.h"
# include "../include/daala/daalaenc.h"
# include "state.h"
# include "entenc.h"
# include "block_size_enc.h"

/*Constants for the packet state machine specific to the encoder.*/
/*No packet currently ready to output.*/
# define OD_PACKET_EMPTY       (0)
/*A packet ready to output.*/
# define OD_PACKET_READY       (1)
/*The number of fractional bits of precision in our \lambda values.*/
# define OD_LAMBDA_SCALE       (2)
/*The number of bits of precision to add to distortion values to match
   \lambda*R.*/
# define OD_ERROR_SCALE        (OD_LAMBDA_SCALE + OD_BITRES)

/*The complexity setting where we enable a square pattern in basic (fullpel)
   MV refinement.*/
# define OD_MC_SQUARE_REFINEMENT_COMPLEXITY (8)
/*The complexity setting where we enable logarithmic (telescoping) MV
   refinement.*/
# define OD_MC_LOGARITHMIC_REFINEMENT_COMPLEXITY (9)
/*The complexity setting where we switch to a square pattern in subpel
   refinement.*/
# define OD_MC_SQUARE_SUBPEL_REFINEMENT_COMPLEXITY (10)

struct od_enc_opt_vtbl {
  int32_t (*mc_compute_sad_4x4)(const unsigned char *src,
   int systride, const unsigned char *ref, int dystride);
  int32_t (*mc_compute_sad_8x8)(const unsigned char *src,
   int systride, const unsigned char *ref, int dystride);
  int32_t (*mc_compute_sad_16x16)(const unsigned char *src,
   int systride, const unsigned char *ref, int dystride);
  int32_t (*mc_compute_sad_32x32)(const unsigned char *src,
   int systride, const unsigned char *ref, int dystride);
  int32_t (*mc_compute_sad_64x64)(const unsigned char *src,
   int systride, const unsigned char *ref, int dystride);
  int32_t (*mc_compute_satd_4x4)(const unsigned char *src,
   int systride, const unsigned char *ref, int dystride);
  int32_t (*mc_compute_satd_8x8)(const unsigned char *src,
   int systride, const unsigned char *ref, int dystride);
  int32_t (*mc_compute_satd_16x16)(const unsigned char *src,
   int systride, const unsigned char *ref, int dystride);
  int32_t (*mc_compute_satd_32x32)(const unsigned char *src,
   int systride, const unsigned char *ref, int dystride);
  int32_t (*mc_compute_satd_64x64)(const unsigned char *src,
   int systride, const unsigned char *ref, int dystride);
};

/*Unsanitized user parameters*/
struct od_params_ctx {
  /*Set using OD_SET_MV_LEVEL_MIN*/
  int mv_level_min;
  /*Set using OD_SET_MV_LEVEL_MAX*/
  int mv_level_max;
};

struct daala_enc_ctx{
  od_state state;
  od_enc_opt_vtbl opt_vtbl;
  oggbyte_buffer obb;
  od_ec_enc ec;
  int packet_state;
  int quality[OD_NPLANES_MAX];
  int complexity;
  int use_activity_masking;
  int use_dering;
  int use_satd;
  int qm;
  int use_haar_wavelet;
  int b_frames;
  od_mv_est_ctx *mvest;
  od_params_ctx params;
#if defined(OD_ENCODER_CHECK)
  struct daala_dec_ctx *dec;
#endif
#if defined(OD_DUMP_BSIZE_DIST)
  /* per frame */
  double bsize_dist[OD_NPLANES_MAX];
  /* per encoder lifetime */
  double bsize_dist_total[OD_NPLANES_MAX];
  FILE *bsize_dist_file;
#endif
  od_block_size_comp *bs;
  /* These buffers are for saving pixel data during block size RDO. */
  od_coeff mc_orig[OD_NBSIZES-1][OD_BSIZE_MAX*OD_BSIZE_MAX];
  od_coeff c_orig[OD_NBSIZES-1][OD_BSIZE_MAX*OD_BSIZE_MAX];
  od_coeff nosplit[OD_NBSIZES-1][OD_BSIZE_MAX*OD_BSIZE_MAX];
  od_coeff split[OD_NBSIZES-1][OD_BSIZE_MAX*OD_BSIZE_MAX];
  od_coeff block_c_orig[OD_BSIZE_MAX*OD_BSIZE_MAX];
  od_coeff block_mc_orig[OD_BSIZE_MAX*OD_BSIZE_MAX];
  od_coeff block_c_noskip[OD_BSIZE_MAX*OD_BSIZE_MAX];
  /* Buffer for the input frame, scaled to reference resolution. */
  od_img input_img[1 + OD_MAX_B_FRAMES];
  unsigned char *input_img_data;
  /** Frame delay. */
  int frame_delay;
  /** Frame counter in encoding order. */
  int64_t enc_order_count;
  /** Frame counter in displaying order. */
  int64_t display_order_count;
  /** Displaying order of current frame being encoded. */
  int64_t curr_display_order;
  /** Current input frame pointer of in_imgs[]. */
  int curr_frame;
  /** Tail pointer of in_imgs[]. */
  int in_buff_ptr;
  /** Head pointer of in_imgs[]. */
  int in_buff_head;
  /** # of frames left in buffer to encode. */
  int frames_in_buff;
  /** Keep the display order of frames in input image buffer. */
  int in_imgs_id[1 + OD_MAX_B_FRAMES];
  /** Number of I or P frames encoded so far, starting from zero. */
  unsigned int ip_frame_count;
#if defined(OD_DUMP_IMAGES) || defined(OD_DUMP_RECONS)
  unsigned char *output_img_data;
  /** Output images buffer, used as circular queue. */
  od_img output_img[2];
  /** Tail pointer of out_imgs[]. */
  int out_buff_ptr;
  /** Head pointer of out_imgs[]. */
  int out_buff_head;
  /** Current decoded frame pointer of out_imgs[]. */
  int curr_dec_frame;
  /** Current output frame pointer of out_imgs[]. */
  int curr_dec_output;
  /** # of frames left in output buffer to display. */
  int frames_in_out_buff;
  /** Keep the display order of frames in output image buffers. */
  int out_imgs_id[2];
#endif
#if defined(OD_DUMP_IMAGES)
  od_img vis_img;
  od_img tmp_vis_img;
  unsigned char *upsample_line_buf[8];
# if defined(OD_ANIMATE)
  int ani_iter;
# endif
#endif
};

/** Holds important encoder information so we can roll back decisions */
struct od_rollback_buffer {
  od_ec_enc ec;
  od_adapt_ctx adapt;
};

void od_encode_checkpoint(const daala_enc_ctx *enc, od_rollback_buffer *rbuf);
void od_encode_rollback(daala_enc_ctx *enc, const od_rollback_buffer *rbuf);

od_mv_est_ctx *od_mv_est_alloc(od_enc_ctx *enc);
void od_mv_est_free(od_mv_est_ctx *est);
void od_mv_est(od_mv_est_ctx *est, int lambda);

int32_t od_mc_compute_sad8_4x4_c(const unsigned char *src, int systride,
 const unsigned char *ref, int dystride);
int32_t od_mc_compute_sad8_8x8_c(const unsigned char *src, int systride,
 const unsigned char *ref, int dystride);
int32_t od_mc_compute_sad8_16x16_c(const unsigned char *src, int systride,
 const unsigned char *ref, int dystride);
int32_t od_mc_compute_sad8_32x32_c(const unsigned char *src, int systride,
 const unsigned char *ref, int dystride);
int32_t od_mc_compute_sad8_64x64_c(const unsigned char *src, int systride,
 const unsigned char *ref, int dystride);
int32_t od_mc_compute_sad8_c(const unsigned char *src, int systride,
 const unsigned char *ref, int dystride, int w, int h);
int32_t od_mc_compute_satd8_4x4_c(const unsigned char *src, int systride,
 const unsigned char *ref, int dystride);
int32_t od_mc_compute_satd8_8x8_c(const unsigned char *src, int systride,
 const unsigned char *ref, int dystride);
int32_t od_mc_compute_satd8_16x16_c(const unsigned char *src, int systride,
 const unsigned char *ref, int dystride);
int32_t od_mc_compute_satd8_32x32_c(const unsigned char *src, int systride,
 const unsigned char *ref, int dystride);
int32_t od_mc_compute_satd8_64x64_c(const unsigned char *src, int systride,
 const unsigned char *ref, int dystride);
int32_t od_mc_compute_sad16_4x4_c(const unsigned char *src, int systride,
 const unsigned char *ref, int dystride);
int32_t od_mc_compute_sad16_8x8_c(const unsigned char *src, int systride,
 const unsigned char *ref, int dystride);
int32_t od_mc_compute_sad16_16x16_c(const unsigned char *src, int systride,
 const unsigned char *ref, int dystride);
int32_t od_mc_compute_sad16_32x32_c(const unsigned char *src, int systride,
 const unsigned char *ref, int dystride);
int32_t od_mc_compute_sad16_64x64_c(const unsigned char *src, int systride,
 const unsigned char *ref, int dystride);
int32_t od_mc_compute_sad16_c(const unsigned char *src, int systride,
 const unsigned char *ref, int dystride, int w, int h);
int32_t od_mc_compute_satd16_4x4_c(const unsigned char *src, int systride,
 const unsigned char *ref, int dystride);
int32_t od_mc_compute_satd16_8x8_c(const unsigned char *src, int systride,
 const unsigned char *ref, int dystride);
int32_t od_mc_compute_satd16_16x16_c(const unsigned char *src, int systride,
 const unsigned char *ref, int dystride);
int32_t od_mc_compute_satd16_32x32_c(const unsigned char *src, int systride,
 const unsigned char *ref, int dystride);
int32_t od_mc_compute_satd16_64x64_c(const unsigned char *src, int systride,
 const unsigned char *ref, int dystride);
void od_enc_opt_vtbl_init_c(od_enc_ctx *enc);

# if defined(OD_DUMP_IMAGES)
void od_encode_fill_vis(daala_enc_ctx *enc);
void od_img_draw_line(od_img *img, int x0, int y0, int x1, int y1,
 const unsigned char ycbcr[3]);
void od_state_draw_mvs(daala_enc_ctx *enc);
# endif

# if defined(OD_X86ASM)
void od_enc_opt_vtbl_init_x86(od_enc_ctx *enc);
# endif

#endif
