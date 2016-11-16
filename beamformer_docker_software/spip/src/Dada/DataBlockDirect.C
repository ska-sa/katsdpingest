/***************************************************************************
 *
 *   Copyright (C) 2015 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/DataBlockDirect.h"

using namespace std;

spip::DataBlockDirect::DataBlockDirect (const char * key_string)
{
  // create a new data block
  db = new DataBlock (key_string);
  db->connect(); 

  is_block_open = false;
  curr_block_id = 0;
  curr_block_ptr = 0;
  curr_block_bytes = 0;
}

spip::DataBlockDirect::~DataBlockDirect ()
{
}
