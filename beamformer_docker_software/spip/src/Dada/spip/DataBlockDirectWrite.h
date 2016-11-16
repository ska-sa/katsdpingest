
#ifndef __DataBlockDirectWrite_h
#define __DataBlockDirectWrite_h

#include "spip/DataBlockDirect.h"

#include <cstddef>
#include <string>

namespace spip {

  class DataBlockDirectWrite : public DataBlockDirect {

    public:

      DataBlockDirectWrite (const char * key);

      ~DataBlockDirectWrite ();

      void * open_block ();

      void close_block (uint64_t bytes_used);

      void add_bytes_written (uint64_t bytes);

    protected:

    private:

  };

}

#endif
