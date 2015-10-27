/******************************************************************************
 * NLK - Neural Language Kit
 *
 * Copyright (c) 2015 Luis Rei <me@luisrei.com> http://luisrei.com @lmrei
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to 
 * deal in the Software without restriction, including without limitation the 
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/


/** @file nlk_string.c
 * Unicode String Manipulation
 */

#include <string.h>
#include <locale.h>
#include <langinfo.h>

#include "nlk_err.h"

#include "nlk_string.h"


/**
 * Convert a (NULL terminated) string to lower case (case folding).
 *
 * @param src   string to convert to lower case (utf8)
 * @param n     maximum number of **bytes** to convert (in destination) or <= 0
 * @param dst   lower case string will be written to this address (utf8)
 * @return  length of string - strlen() - in bytes after conversion or -1 
 *
 * Unicode lower case and upper case characters do not necessarily have the 
 * same number of bytes when represented in (multibyte) utf8. Thus src and 
 * dst do not have necessarily the same size in bytes.
 *
 * This function handles the conversion to lower case and if successful returns
 * the length in bytes of the converted string which is stored in *dst*. This
 * lenght is the same as calling strlen(dst): the number of bytes that precede 
 * the NULL terminator.
 *
 * The parameter *n* exists to guard against the possibility that an
 * insuficiently sized *dst* char (byte) array was passed as a parameter.
 * To succeed this function requires that *n* be equal or greater to the 
 * resulting bytes + 1 (for the NULL terminator). 
 *
 * If the value passed as *n* is less or equal to 0, this check will be ignored 
 * and an overflow can occur.
 *
 * *dst* will always be NULL terminated if the function returns >= 0
 * If the function returns < 0, an error has occurred and *dst* is unusable.
 */
ssize_t
nlk_string_lower(const char *src, const ssize_t n, char *dst)
{
    /* int32 chars (unicode) */
    utf8proc_int32_t src_ch;    /**< source character as an int32 */
    utf8proc_int32_t dst_ch;    /**< lower case version as an int32 */

    /* lengths/positions */
    ssize_t src_pos = 0;    /**< current position in the source string */
    ssize_t len = 0;        /**< length of dst after (and during) conversion */

    /* return values and temporary memory */
    utf8proc_ssize_t bytes = 0;      /**< bytes read/written for current character */
    uint8_t buf[4];         /**< temporary buf for int32 -> char array */

    /* utf8proc expects "chars" to be unsigned so let's just do that now
     * in order to avoid casts everywhere */
    const utf8proc_uint8_t *src_u = (utf8proc_uint8_t *) src;
    utf8proc_uint8_t *dst_u = (utf8proc_uint8_t *) dst;


    /* we iterate through our source string, converting the string to lower 
     * case one character at a time */
    while(src[src_pos] != '\0') {
        /* convert each character to 32bit unicode value representation 
         * (bytes is the number of bytes read) */
        bytes = utf8proc_iterate(&src_u[src_pos], -1, &src_ch);
        nlk_assert_silent(bytes > 0);   /* Failure */

        /* unicode values in a char can be multiple bytes, thus: */
        src_pos += bytes;

        /* convert to lower case */
        dst_ch = utf8proc_tolower(src_ch);

        /* check boundaries to make sure destination won't overflown */
        if(n > 0) {
            /* write unicode character to the buffer as a utf8 byte sequence
             * (bytes is the number of bytes written) */
            bytes = utf8proc_encode_char(dst_ch, buf);
            nlk_assert_silent(bytes > 0);   /* Failure */

            nlk_assert_silent(len + bytes < n - 1);

            /* copy to destination */
            stpncpy(&dst[len], (char *)buf, bytes);
        } else {
            /* I too like to live dangerously - in case the *n* parameter was
             * <= 0 we do a direct conversion and hope for the best...
             */
            bytes = utf8proc_encode_char(dst_ch, &dst_u[len]);

        }

        /* a unicode character might be converted to several chars, thus: */
        len += bytes;
    }

    /* null terminate our pretty new string */
    dst[len] = '\0';

    return len;

error:
    return -1;
}


/**
 * Convert a (NULL terminated) string to upper case.
 *
 * @param src   string to convert to upper case (utf8)
 * @param n     maximum number of **bytes** to convert (in destination) or <= 0
 * @param dst   upper case string will be written to this address (utf8)
 * @return  length of string - strlen() - in bytes after conversion or -1 
 *
 * Unicode upper case and upper case characters do not necessarily have the 
 * same number of bytes when represented in (multibyte) utf8. Thus src and 
 * dst do not have necessarily the same size in bytes.
 *
 * This function handles the conversion to upper case and if successful returns
 * the length in bytes of the converted string which is stored in *dst*. This
 * lenght is the same as calling strlen(dst): the number of bytes that precede 
 * the NULL terminator.
 *
 * The parameter *n* exists to guard against the possibility that an
 * insuficiently sized *dst* char (byte) array was passed as a parameter.
 * To succeed this function requires that *n* be equal or greater to the 
 * resulting bytes + 1 (for the NULL terminator). 
 *
 * If the value passed as *n* is less or equal to 0, this check will be ignored 
 * and an overflow can occur.
 *
 * *dst* will always be NULL terminated if the function returns >= 0
 * If the function returns < 0, an error has occurred and *dst* is unusable.
 */
ssize_t
nlk_string_upper(const char *src, const ssize_t n, char *dst)
{
    /* int32 chars (unicode) */
    utf8proc_int32_t src_ch;    /**< source character as an int32 */
    utf8proc_int32_t dst_ch;    /**< upper case version as an int32 */

    /* lengths/positions */
    ssize_t src_pos = 0;    /**< current position in the source string */
    ssize_t len = 0;        /**< length of dst after (and during) conversion */

    /* return values and temporary memory */
    utf8proc_ssize_t bytes = 0;      /**< bytes r/w for current character */
    uint8_t buf[4];         /**< temporary buf for int32 -> char array */

    /* utf8proc expects "chars" to be unsigned so let's just do that now
     * in order to avoid casts everywhere */
    const utf8proc_uint8_t *src_u = (utf8proc_uint8_t *) src;
    utf8proc_uint8_t *dst_u = (utf8proc_uint8_t *) dst;


    /* we iterate through our source string, converting the string to upper 
     * case one character at a time */
    while(src[src_pos] != '\0') {
        /* convert each character to 32bit unicode value representation 
         * (bytes is the number of bytes read) */
        bytes = utf8proc_iterate(&src_u[src_pos], -1, &src_ch);
        nlk_assert_silent(bytes > 0);   /* Failure */

        /* unicode values in a char can be multiple bytes, thus: */
        src_pos += bytes;

        /* convert to upper case */
        dst_ch = utf8proc_toupper(src_ch);

        /* check boundaries to make sure destination won't overflown */
        if(n > 0) {
            /* write unicode character to the buffer as a utf8 byte sequence
             * (bytes is the number of bytes written) */
            bytes = utf8proc_encode_char(dst_ch, buf);
            nlk_assert_silent(bytes > 0);   /* Failure */

            nlk_assert_silent(len + bytes < n - 1);

            /* copy to destination */
            stpncpy(&dst[len], (char *)buf, bytes);
        } else {
            /* I too like to live dangerously - in case the *n* parameter was
             * <= 0 we do a direct conversion and hope for the best...
             */
            bytes = utf8proc_encode_char(dst_ch, &dst_u[len]);

        }

        /* a unicode character might be converted to several chars, thus: */
        len += bytes;
    }

    /* null terminate our pretty new string */
    dst[len] = '\0';

    return len;

error:
    return -1;

}

/**
 * Case fold a string (lowercase result)
 *
 * @param src   string to convert to upper case (utf8)
 * @param n     maximum number of **bytes** to convert (in destination) or <= 0
 * @param dst   lower case string will be written to this address (utf8)
 * @return length of string - strlen() - in bytes after conversion or -1 
 *
 * Unicode upper case and upper case characters do not necessarily have the 
 * same number of bytes when represented in (multibyte) utf8. Thus src and 
 * dst do not have necessarily the same size in bytes.
 *
 * This function handles case folding and if successful returns
 * the length in bytes of the converted string which is stored in *dst*. This
 * lenght is the same as calling strlen(dst): the number of bytes that precede 
 * the NULL terminator.
 *
 * The parameter *n* exists to guard against the possibility that an
 * insuficiently sized *dst* char (byte) array was passed as a parameter.
 * To succeed this function requires that *n* be equal or greater to the 
 * resulting bytes + 1 (for the NULL terminator). 
 *
 * If the value passed as *n* is less or equal to 0, this check will be ignored 
 * and an overflow can occur.
 *
 * *dst* will always be NULL terminated if the function returns >= 0
 * If the function returns < 0, an error has occurred and *dst* is unusable.

 *
 */
ssize_t
nlk_string_case_fold(const char *src, const ssize_t n, char *dst)
{
    /* int32 chars (unicode) */
    utf8proc_int32_t src_ch;    /**< source character as an int32 */
    utf8proc_int32_t dst_ch;    /**< case folded version as an int32 */

    /* lengths/positions */
    ssize_t src_pos = 0;    /**< current position in the source string */
    ssize_t len = 0;        /**< length of dst after (and during) conversion */

    /* return values and temporary memory */
    utf8proc_ssize_t bytes = 0;      /**< bytes r/w for current character */
    uint8_t buf[4];         /**< temporary buf for int32 -> char array */

    /* utf8proc expects "chars" to be unsigned so let's just do that now
     * in order to avoid casts everywhere */
    const utf8proc_uint8_t *src_u = (utf8proc_uint8_t *) src;
    utf8proc_uint8_t *dst_u = (utf8proc_uint8_t *) dst;


    /* we iterate through our source string, converting the string to lower 
     * case, than upper case, than back to lower case one character at a time 
     *
     * This lower-upper-lower is necessary because unicode is a bit retarded.
     * */
    while(src[src_pos] != '\0') {
        /* convert each character to 32bit unicode value representation 
         * (bytes is the number of bytes read) */
        bytes = utf8proc_iterate(&src_u[src_pos], -1, &src_ch);
        nlk_assert_silent(bytes > 0);   /* Failure */

        /* unicode values in a char can be multiple bytes, thus: */
        src_pos += bytes;

        /* first convert to lower case */
        dst_ch = utf8proc_tolower(src_ch);
        /* now convert to upper case */
        dst_ch = utf8proc_toupper(dst_ch);
        /* finally back to lower case */
        dst_ch = utf8proc_tolower(dst_ch);

        /* check boundaries to make sure destination won't overflown */
        if(n > 0) {
            /* write unicode character to the buffer as a utf8 byte sequence
             * (bytes is the number of bytes written) */
            bytes = utf8proc_encode_char(dst_ch, buf);
            nlk_assert_silent(bytes > 0);   /* Failure */

            nlk_assert_silent(len + bytes < n - 1);

            /* copy to destination */
            stpncpy(&dst[len], (char *)buf, bytes);
        } else {
            /* I too like to live dangerously - in case the *n* parameter was
             * <= 0 we do a direct conversion and hope for the best...
             */
            bytes = utf8proc_encode_char(dst_ch, &dst_u[len]);
        }

        /* a unicode character might be converted to several chars, thus: */
        len += bytes;
    }

    /* null terminate our pretty new string */
    dst[len] = '\0';

    return len;

error:
    return -1;

}

/**
 * Reads the UTF8 character in src at position *pos* and writes it to string
 * *dst*. If successful the string will be NULL terminated and it's length 
 * in bytes up to the NULL terminator is returned.
 *
 * @param src   source string to get the character from
 * @param pos   position in bytes from which the character will be extracted
 * @param dst   destination to where the character will be written
 * @return bytes read from src and written to dst or < 0 if an error occurred.
 *
 * @note
 * *dst* should be at least 5 bytes (4 bytes for a unicode character +1 NULL).
 * @endnote
 */
ssize_t
nlk_string_get_char(const char *src, size_t pos, char *dst)
{
    ssize_t bytes;          /**< bytes read from src and written to dst */
    utf8proc_int32_t ch;

    /* convert character to int32 */
    bytes = utf8proc_iterate((const utf8proc_uint8_t *) &src[pos], -1, &ch);
    nlk_assert_silent(bytes > 0);   /* Failure */

    /* convert back to utf8 multibyte char */
    bytes = utf8proc_encode_char(ch, (utf8proc_uint8_t *) dst);
    nlk_assert_silent(bytes > 0);   /* Failure */

    /* null terminate our new string */
    dst[bytes] = '\0';

    return bytes;

error:
    return -1;
}


/**
 * True if current locale is utf8
 */
bool
nlk_string_is_locale_utf8()
{
    return strcmp(nl_langinfo(CODESET), "UTF-8") == 0;
}
