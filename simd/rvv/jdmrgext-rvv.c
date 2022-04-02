/*
 * Risc-V vector extension optimizations for libjpeg-turbo.
 *
 * Copyright (C) 2020, Arm Limited.  All Rights Reserved.
 * Copyright (C) 2020, D. R. Commander.  All Rights Reserved.
 *
 * This software is provided 'as-is', without any express or implied
 * warranty.  In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgment in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 */

/* This file is included by jdmerge-rvv.c. */

/* These routines combine simple (non-fancy, i.e. non-smooth) h2v1 or h2v2
 * chroma upsampling and YCbCr -> RGB color conversion into a single function.
 *
 * As with the standalone functions, YCbCr -> RGB conversion is defined by the
 * following equations:
 *    R = Y                        + 1.40200 * (Cr - 128)
 *    G = Y - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128)
 *    B = Y + 1.77200 * (Cb - 128)
 *
 * Scaled integer constants are used to avoid floating-point arithmetic:
 *    0.3441467 = 11277 * 2^-15
 *    0.7141418 = 23401 * 2^-15
 *    1.4020386 = 22971 * 2^-14
 *    1.7720337 = 29033 * 2^-14
 * These constants are defined in jdmerge-neon.c.
 *
 * To ensure correct results, rounding is used when descaling.
 */


void jsimd_h2v1_merged_upsample_rvv(JDIMENSION output_width,
                                    JSAMPIMAGE input_buf,
                                    JDIMENSION in_row_group_ctr,
                                    JSAMPARRAY output_buf)
{
    JSAMPROW outptr;
    /* Pointers to Y, Cb, and Cr data */
    JSAMPROW inptr0, inptr1, inptr2;
    int pitch = output_width * RGB_PIXELSIZE, num_cols, yloop;
#if BITS_IN_JSAMPLE == 8
    vuint8m1_t dest, src;
#endif
    size_t vl, cols;
    vl = vsetvl_e16m2(output_width / 2);
    ptrdiff_t bstride;
    vuint16m2_t y0, y1, cb, cr, r_sub_y, y_sub_g, b_sub_y, r, g, b, tmp;

    /* Constants. */
#if RGB_PIXELSIZE == 4
#if BITS_IN_JSAMPLE == 8
    uint8_t alpha[1] = { 0xFF };
    vuint8m1_t alpha_v = vlse8_v_u8m1(alpha, 0, vl);
#else           /* BITS_IN_JSAMPLE == 12 */
    uint16_t alpha[1] = { 0xFF };
    vuint16m2_t alpha_v = vlse16_v_u16m2(alpha, 0, vl);
#endif
#endif

    inptr0 = input_buf[0][in_row_group_ctr];
    inptr1 = input_buf[1][in_row_group_ctr];
    inptr2 = input_buf[2][in_row_group_ctr];
    outptr = output_buf[0];

    /* TODO: find a way to deal with the situation when output_width is odd value. */
    for (num_cols = pitch; num_cols > 0; 
         inptr0 += 2 * vl, inptr1 += vl, inptr2 += vl,
         outptr += 2 * cols, num_cols -= 2 * cols) {
        /* Set vl for each iteration. */
        vl = vsetvl_e16m2(num_cols / RGB_PIXELSIZE / 2);
        cols = vl * RGB_PIXELSIZE;
        bstride = RGB_PIXELSIZE * sizeof(JSAMPLE);

        /* Load R, G, B channels as vectors from inptr. */
#if BITS_IN_JSAMPLE == 8
        /* Extending to vuint16m4_t type for following multiply calculation. */
        /* Y component values with even-numbered indices. */
        src = vlse8_v_u8m1(inptr0, 2 * sizeof(JSAMPLE), vl);
        y0 = vwaddu_vx_u16m2(src, 0, vl); /* Widening to vuint16m4_t type */
        /* Y component values with odd-numbered indices. */
        src = vlse8_v_u8m1(inptr0 + 1, 2 * sizeof(JSAMPLE), vl);
        y1 = vwaddu_vx_u16m2(src, 0, vl); /* Widening to vuint16m4_t type */
        src = vle8_v_u8m1(inptr1, vl);
        cb = vwaddu_vx_u16m2(src, 0, vl); /* Widening to vuint16m4_t type */
        src = vle8_v_u8m1(inptr2, vl);
        cr = vwaddu_vx_u16m2(src, 0, vl); /* Widening to vuint16m4_t type */
#else                                     /* BITS_IN_JSAMPLE == 12 */
        y0 = vlse16_v_u16m2(inptr0, 2 * sizeof(JSAMPLE), vl);
        y1 = vlse16_v_u16m2(inptr0 + 1, 2 * sizeof(JSAMPLE), vl);
        cb = vle16_v_u16m2(inptr1, vl);
        cr = vle16_v_u16m2(inptr2, vl);
#endif

        /* (Original)
         * R = Y                + 1.40200 * (Cr - CENTERJSAMPLE)
         * G = Y - 0.34414 * (Cb - CENTERJSAMPLE) - 0.71414 * (Cr - CENTERJSAMPLE)
         * B = Y + 1.77200 * (Cb - CENTERJSAMPLE)
         *
         * (This implementation)
         * R = Y                + 0.40200 * (Cr - CENTERJSAMPLE) + (Cr - CENTERJSAMPLE)
         * G = Y - 0.34414 * (Cb - CENTERJSAMPLE) - 0.71414 * (Cr - CENTERJSAMPLE)
         * B = Y + 0.77200 * (Cb - CENTERJSAMPLE) + (Cb - CENTERJSAMPLE)
         * (Because 16-bit can only represent values < 1.)
         */
        cb = vsub_vx_u16m2(cb, CENTERJSAMPLE, vl);  /* Cb - CENTERJSAMPLE */
        cr = vsub_vx_u16m2(cr, CENTERJSAMPLE, vl);  /* Cr - CENTERJSAMPLE */

        /* Calculate R-Y values */
        r_sub_y = vmulhu_vx_u16m2(cr, F_0_402, vl);
        r_sub_y = vadd_vv_u16m2(r_sub_y, cr, vl);

        /* Calculate Y-G values */
        y_sub_g = vmulhu_vx_u16m2(cb, F_0_344, vl);
        tmp = vmulhu_vx_u16m2(cr, F_0_714, vl);
        y_sub_g = vadd_vv_u16m2(y_sub_g, tmp, vl);

        /* Calculate B-Y values */
        b_sub_y = vmulhu_vx_u16m2(cb, F_0_772, vl);
        b_sub_y = vadd_vv_u16m2(b_sub_y, cb, vl);

        /* Compute R, G, B values with even-numbered indices. */
        r = vadd_vv_u16m2(r_sub_y, y0, vl);
        g = vsub_vv_u16m2(y0, y_sub_g, vl);
        b = vadd_vv_u16m2(b_sub_y, y0, vl);
        /* Narrow to 8-bit and store to memory. */
#if BITS_IN_JSAMPLE == 8
        dest = vnsrl_wx_u8m1(r, 0, vl);     /* Narrowing from 16-bit to 8-bit. */
        vsse8_v_u8m1(outptr + RGB_RED, 2 * bstride, dest, vl);
        dest = vnsrl_wx_u8m1(g, 0, vl);     /* Narrowing from 16-bit to 8-bit. */
        vsse8_v_u8m1(outptr + RGB_GREEN, 2 * bstride, dest, vl);
        dest = vnsrl_wx_u8m1(b, 0, vl);     /* Narrowing from 16-bit to 8-bit. */
        vsse8_v_u8m1(outptr + RGB_BLUE, 2 * bstride, dest, vl);
#else   /* BITS_IN_JSAMPLE == 12 */
        vsse16_v_u16m2(outptr + RGB_RED, 2 * bstride, r, vl);
        vsse16_v_u16m2(outptr + RGB_GREEN, 2 * bstride, g, vl);
        vsse16_v_u16m2(outptr + RGB_BLUE, 2 * bstride, b, vl);
#endif
        /* Deal with alpha channel. */
#if RGB_PIXELSIZE == 4
#if BITS_IN_JSAMPLE == 8
        vsse8_v_u8m1(outptr + RGB_ALPHA, 2 * bstride, alpha_v, vl);
#else           /* BITS_IN_JSAMPLE == 12 */
        vsse16_v_u16m2(outptr + RGB_ALPHA, 2 * bstride, alpha_v, vl);
#endif
#endif

        /* Compute R, G, B values with odd-numbered indices. */
        r = vadd_vv_u16m2(r_sub_y, y1, vl);
        g = vsub_vv_u16m2(y1, y_sub_g, vl);
        b = vadd_vv_u16m2(b_sub_y, y1, vl);
        /* Narrow to 8-bit and store to memory. */
#if BITS_IN_JSAMPLE == 8
        dest = vnsrl_wx_u8m1(r, 0, vl);     /* Narrowing from 16-bit to 8-bit. */
        vsse8_v_u8m1(outptr + RGB_PIXELSIZE + RGB_RED, 2 * bstride, dest, vl);
        dest = vnsrl_wx_u8m1(g, 0, vl);     /* Narrowing from 16-bit to 8-bit. */
        vsse8_v_u8m1(outptr + RGB_PIXELSIZE + RGB_GREEN, 2 * bstride, dest, vl);
        dest = vnsrl_wx_u8m1(b, 0, vl);     /* Narrowing from 16-bit to 8-bit. */
        vsse8_v_u8m1(outptr + RGB_PIXELSIZE + RGB_BLUE, 2 * bstride, dest, vl);
#else   /* BITS_IN_JSAMPLE == 12 */
        vsse16_v_u16m2(outptr + RGB_PIXELSIZE + RGB_RED, 2 * bstride, r, vl);
        vsse16_v_u16m2(outptr + RGB_PIXELSIZE + RGB_GREEN, 2 * bstride, g, vl);
        vsse16_v_u16m2(outptr + RGB_PIXELSIZE + RGB_BLUE, 2 * bstride, b, vl);
#endif
        /* Deal with alpha channel. */
#if RGB_PIXELSIZE == 4
#if BITS_IN_JSAMPLE == 8
        vsse8_v_u8m1(outptr + RGB_PIXELSIZE + RGB_ALPHA, 2 * bstride, alpha_v, vl);
#else           /* BITS_IN_JSAMPLE == 12 */
        vsse16_v_u16m2(outptr + RGB_PIXELSIZE + RGB_ALPHA, 2 * bstride, alpha_v, vl);
#endif
#endif

    }
}


void jsimd_h2v2_merged_upsample_rvv(JDIMENSION output_width,
                                    JSAMPIMAGE input_buf,
                                    JDIMENSION in_row_group_ctr,
                                    JSAMPARRAY output_buf)
{
    JSAMPROW inptr, outptr;

    inptr = input_buf[0][in_row_group_ctr];
    outptr = output_buf[0];

    input_buf[0][in_row_group_ctr] = input_buf[0][in_row_group_ctr * 2];
    jsimd_h2v1_merged_upsample_rvv(output_width, input_buf, in_row_group_ctr,
                                   output_buf);

    input_buf[0][in_row_group_ctr] = input_buf[0][in_row_group_ctr * 2 + 1];
    output_buf[0] = output_buf[1];
    jsimd_h2v1_merged_upsample_rvv(output_width, input_buf, in_row_group_ctr,
                                   output_buf);

    input_buf[0][in_row_group_ctr] = inptr;
    output_buf[0] = outptr;
}