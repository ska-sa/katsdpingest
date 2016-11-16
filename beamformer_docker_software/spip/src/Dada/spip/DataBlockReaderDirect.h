
#ifndef __DataBlock_h
#define __DataBlock_h

#include "ipcio.h"
#include "ipcbuf.h"

#include <cstddef>
#include <string>

namespace spip {

  class DataBlock {

    public:

      DataBlock (const char * key);

      ~DataBlock ();

      void * open_block ();

      void close_block ();

      bool block_full ();

      void add_bytes_written (uint64_t bytes_to_add);

      void * open_block;




    protected:

    private:

      DataBlock * db;

      block_open


  };

}

#endif
