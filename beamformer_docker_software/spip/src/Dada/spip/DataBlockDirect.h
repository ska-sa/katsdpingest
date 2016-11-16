
#ifndef __DataBlockDirect_h
#define __DataBlockDirect_h

#include "spip/DataBlock.h"

#include <cstddef>

namespace spip {

  class DataBlockDirect {

    public:

      DataBlockDirect (const char * key);

      ~DataBlockDirect ();

      virtual void * open_block () = 0;

      virtual void close_block (uint64_t bytes_used) = 0;

      inline bool block_full () { return (curr_block_bytes == curr_block_size); };

      bool block_open () { return is_block_open; };

      void * get_block_ptr () { return curr_block_ptr; };

      uint64_t get_block_id () { return curr_block_id; };

    protected:

      DataBlock * db;

      // flag for currently open data block
      bool is_block_open;

      // currently open data block id
      uint64_t curr_block_id;

      void * curr_block_ptr;

      uint64_t curr_block_bytes;

      uint64_t curr_block_size;

    private:

  };

}

#endif
