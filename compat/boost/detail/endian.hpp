#ifndef TARTANCALIB_COMPAT_BOOST_DETAIL_ENDIAN_HPP
#define TARTANCALIB_COMPAT_BOOST_DETAIL_ENDIAN_HPP

#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && \
    (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
#define BOOST_BIG_ENDIAN
#elif defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && \
    (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#define BOOST_LITTLE_ENDIAN
#elif defined(_WIN32)
#define BOOST_LITTLE_ENDIAN
#else
#error "Unable to determine platform endianness for Boost compatibility shim."
#endif

#endif  // TARTANCALIB_COMPAT_BOOST_DETAIL_ENDIAN_HPP
