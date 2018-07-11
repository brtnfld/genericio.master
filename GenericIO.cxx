/*
 *                    Copyright (C) 2015, UChicago Argonne, LLC
 *                               All Rights Reserved
 * 
 *                               Generic IO (ANL-15-066)
 *                     Hal Finkel, Argonne National Laboratory
 * 
 *                              OPEN SOURCE LICENSE
 * 
 * Under the terms of Contract No. DE-AC02-06CH11357 with UChicago Argonne,
 * LLC, the U.S. Government retains certain rights in this software.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 *   1. Redistributions of source code must retain the above copyright notice,
 *      this list of conditions and the following disclaimer.
 * 
 *   2. Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 * 
 *   3. Neither the names of UChicago Argonne, LLC or the Department of Energy
 *      nor the names of its contributors may be used to endorse or promote
 *      products derived from this software without specific prior written
 *      permission.
 * 
 * *****************************************************************************
 * 
 *                                  DISCLAIMER
 * THE SOFTWARE IS SUPPLIED “AS IS” WITHOUT WARRANTY OF ANY KIND.  NEITHER THE
 * UNTED STATES GOVERNMENT, NOR THE UNITED STATES DEPARTMENT OF ENERGY, NOR
 * UCHICAGO ARGONNE, LLC, NOR ANY OF THEIR EMPLOYEES, MAKES ANY WARRANTY,
 * EXPRESS OR IMPLIED, OR ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE
 * ACCURACY, COMPLETENESS, OR USEFULNESS OF ANY INFORMATION, DATA, APPARATUS,
 * PRODUCT, OR PROCESS DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT INFRINGE
 * PRIVATELY OWNED RIGHTS.
 * 
 * *****************************************************************************
 */

#define _XOPEN_SOURCE 600
#include "CRC64.h"
#include "GenericIO.h"

extern "C" {
#include "blosc.h"
}

#include <sstream>
#include <fstream>
#include <stdexcept>
#include <iterator>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>

#ifndef GENERICIO_NO_MPI
#include <ctime>
#endif

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>

#ifdef __bgq__
#include <mpix.h>
#endif

#ifndef MPI_UINT64_T
#define MPI_UINT64_T (sizeof(long) == 8 ? MPI_LONG : MPI_LONG_LONG)
#endif

using namespace std;

namespace gio {


#ifndef GENERICIO_NO_MPI
GenericFileIO_MPI::~GenericFileIO_MPI() {
  (void) MPI_File_close(&FH);
}

void GenericFileIO_MPI::open(const std::string &FN, bool ForReading) {
  FileName = FN;

  int amode = ForReading ? MPI_MODE_RDONLY : (MPI_MODE_WRONLY | MPI_MODE_CREATE);
  if (MPI_File_open(Comm, const_cast<char *>(FileName.c_str()), amode,
                    MPI_INFO_NULL, &FH) != MPI_SUCCESS)
    throw runtime_error((!ForReading ? "Unable to create the file: " :
                                       "Unable to open the file: ") +
                        FileName);
}

void GenericFileIO_MPI::setSize(size_t sz) {
  if (MPI_File_set_size(FH, sz) != MPI_SUCCESS)
    throw runtime_error("Unable to set size for file: " + FileName);
}

void GenericFileIO_MPI::read(void *buf, size_t count, off_t offset,
                             const std::string &D) {
  while (count > 0) {
    MPI_Status status;
    if (MPI_File_read_at(FH, offset, buf, count, MPI_BYTE, &status) != MPI_SUCCESS)
      throw runtime_error("Unable to read " + D + " from file: " + FileName);

    int scount;
    (void) MPI_Get_count(&status, MPI_BYTE, &scount);

    count -= scount;
    buf = ((char *) buf) + scount;
    offset += scount;
  }
}

void GenericFileIO_MPI::write(const void *buf, size_t count, off_t offset,
                              const std::string &D) {
  while (count > 0) {
    MPI_Status status;
    if (MPI_File_write_at(FH, offset, (void *) buf, count, MPI_BYTE, &status) != MPI_SUCCESS)
      throw runtime_error("Unable to write " + D + " to file: " + FileName);

    int scount = 0;
    // On some systems, MPI_Get_count will not return zero even when count is zero.
    if (count > 0)
      (void) MPI_Get_count(&status, MPI_BYTE, &scount);

    count -= scount;
    buf = ((char *) buf) + scount;
    offset += scount;
  }
}

void GenericFileIO_MPICollective::read(void *buf, size_t count, off_t offset,
                             const std::string &D) {
  int Continue = 0;

  do {
    MPI_Status status;
    if (MPI_File_read_at_all(FH, offset, buf, count, MPI_BYTE, &status) != MPI_SUCCESS)
      throw runtime_error("Unable to read " + D + " from file: " + FileName);

    int scount = 0;
    // On some systems, MPI_Get_count will not return zero even when count is zero.
    if (count > 0)
      (void) MPI_Get_count(&status, MPI_BYTE, &scount);

    count -= scount;
    buf = ((char *) buf) + scount;
    offset += scount;

    int NeedContinue = (count > 0);
    MPI_Allreduce(&NeedContinue, &Continue, 1, MPI_INT, MPI_SUM, Comm);
  } while (Continue);
}

void GenericFileIO_MPICollective::write(const void *buf, size_t count, off_t offset,
                              const std::string &D) {
  int Continue = 0;

  do {
    MPI_Status status;
    if (MPI_File_write_at_all(FH, offset, (void *) buf, count, MPI_BYTE, &status) != MPI_SUCCESS)
      throw runtime_error("Unable to write " + D + " to file: " + FileName);

    int scount;
    (void) MPI_Get_count(&status, MPI_BYTE, &scount);

    count -= scount;
    buf = ((char *) buf) + scount;
    offset += scount;

    int NeedContinue = (count > 0);
    MPI_Allreduce(&NeedContinue, &Continue, 1, MPI_INT, MPI_SUM, Comm);
  } while (Continue);
}
#endif

#ifdef GENERICIO_HAVE_HDF
GenericFileIO_HDF::~GenericFileIO_HDF() {
  herr_t ret;
  const char *EnvStr = getenv("GENERICIO_USE_HDF");
  if (EnvStr)
    ret=H5Fclose(FH);
}
hid_t GenericFileIO_HDF::get_fileid() {
  return this->FH;
}

size_t GenericFileIO_HDF::get_NumElem() {
  return this->NumElem;
}

  void GenericFileIO_HDF::open(const std::string &FN, bool ForReading) {
  hid_t fid, space, dset, attr, atype, gid, aid;	  // HDF5 file IDs
  hid_t fapl_id;  // File access templates
  herr_t ret;     // Generic return value
  hsize_t     dims[1], adim[1];
  hsize_t     domainsize[1];
  int ndims;
  double timestep[1];
  char notes[] = {"Important notes go here"};
  const char *wdata[9]= {"pid", "mask", "phi", 
		   "vx","vy","vz",
		   "x","y","z"};
  size_t NumEl;   
  /* Groups */
  hid_t  grp_l_id;
  hid_t  grp_p_id;
  hid_t  grp_s_id;
  
  /* Container Version & Read Contexts */
  uint64_t version;
  hid_t    rc_id1, rc_id2, rc_id3, rc_id4, rc_id5, rc_id;
  
  /* Transactions  */
  int commRank;

  MPI_Comm_rank(Comm, &commRank);
  FileName = FN;

//   filetype = H5Tcopy (H5T_C_S1);
//   ret = H5Tset_size (filetype, H5T_VARIABLE);

  // setup file access template with parallel IO access.
  fapl_id = H5Pcreate (H5P_FILE_ACCESS);
  ret = H5Pset_fapl_mpio(fapl_id, Comm, MPI_INFO_NULL);
  FileName = FN;

  cout << "in open" << endl;

  if ( ForReading) {
    if( (fid = H5Fopen(const_cast<char *>(FileName.c_str()),H5F_ACC_RDONLY,fapl_id)) < 0)
      throw runtime_error( ("Unable to open the file: ") + FileName);
  } else {
    if( (fid = H5Fcreate(const_cast<char *>(FileName.c_str()),H5F_ACC_TRUNC,H5P_DEFAULT,fapl_id)) < 0)
      throw runtime_error( ("Unable to create the file: ") + FileName);
  }
  /* Get read context */
  
  FH=fid;

  //  if( (fid = H5Fopen(const_cast<char *>(FileName.c_str()),H5F_ACC_RDWR,fapl_id)) < 0 ) {    
  // Release file-access template
  ret = H5Pclose(fapl_id);

  if ( ForReading ) {
    // get size
    dset  = H5Dopen (fid, "/Variables/id", H5P_DEFAULT);
    space = H5Dget_space (dset);
    ndims = H5Sget_simple_extent_dims (space, dims, NULL);

    ret = H5Dclose (dset);
    ret = H5Sclose (space);

    int commRanks;
    MPI_Comm_size(MPI_COMM_WORLD, &commRanks);

    NumElem = dims[0]/commRanks;

  }

  cout << "done open" << endl;
}

void GenericFileIO_HDF::setSize(size_t sz) {
//   if (MPI_File_set_size(FH, sz) != MPI_SUCCESS)
//     throw runtime_error("Unable to set size for file: " + FileName);
}

void GenericFileIO_HDF::read(void *buf, size_t count, off_t offset,
                             const std::string &D) {
  //  cout << "ERROR GenericFileIO_HDF::read" << endl;
//   while (count > 0) {
//     MPI_Status status;
//     if (MPI_File_read_at(FH, offset, buf, count, MPI_BYTE, &status) != MPI_SUCCESS)
//       throw runtime_error("Unable to read " + D + " from file: " + FileName);

//     int scount;
//     (void) MPI_Get_count(&status, MPI_BYTE, &scount);

//     count -= scount;
//     buf = ((char *) buf) + scount;
//     offset += scount;
//   }
}

void GenericFileIO_HDF::read_hdf(void *buf, size_t count, off_t offset,
				 const std::string &D, hid_t dtype, hsize_t numel) {

  hid_t dataset,file_dataspace;
  char *c_str = new char[D.length() + 1];
  hsize_t dims[1]; // dataspace dim sizes
  herr_t ret;
  dims[0] = numel;

  std::strcpy(c_str, D.c_str());

  // open the dataset collectively
  dataset = H5Dopen2(FH, c_str, H5P_DEFAULT); 

  file_dataspace = H5Dget_space (dataset);

  ret = H5Dread(dataset, dtype,  file_dataspace, file_dataspace,
	    H5P_DEFAULT, buf);

  // release dataspaceID
  H5Sclose (file_dataspace);
  // close dataset collectively
  ret = H5Dclose(dataset);

  cout << "read_hdf aborting" << endl;
  abort();

//   while (count > 0) {
//     MPI_Status status;
//     if (MPI_File_read_at(FH, offset, buf, count, MPI_BYTE, &status) != MPI_SUCCESS)
//       throw runtime_error("Unable to read " + D + " from file: " + FileName);

//     int scount;
//     (void) MPI_Get_count(&status, MPI_BYTE, &scount);

//     count -= scount;
//     buf = ((char *) buf) + scount;
//     offset += scount;
//   }
}

void GenericFileIO_HDF::write(const void *buf, size_t count, off_t offset,
				   const std::string &D) {
}

void GenericFileIO_HDF::write_hdf(const void *buf, size_t count, off_t offset,
				  const std::string &D, hid_t dtype, hsize_t numel, const void *CRC, hid_t gid, uint64_t Totnumel) {

  hid_t sid, dataset,file_dataspace, aid, mem_dataspace, attr, fid, aspace;
  
  hsize_t dims[1]; // dataspace dim sizes
  herr_t ret;
  char *c_str = new char[D.length() + 1];
  char *c_str3 = new char[D.length() + 5];
  hsize_t adim[1];
  hsize_t start[1];			/* for hyperslab setting */
  hsize_t hypercount[1], stride[1];	/* for hyperslab setting */
  hsize_t sizedims[1], adims[1];
  int commRank, commRanks;
  MPI_Comm_rank(MPI_COMM_WORLD, &commRank);
  MPI_Comm_size(MPI_COMM_WORLD, &commRanks); 


  adims[0] = (hsize_t)commRanks;
  dims[0] = (hsize_t)Totnumel;


  cout << "inside hdf_write" << endl;

  // --------------------------
  // Define the dimensions of the overall datasets
  // and create the dataset
  // -------------------------
  // setup dimensionality object

  std::strcpy(c_str, D.c_str());

  if( (sid = H5Screate_simple (1, dims, NULL)) < 0)
      throw runtime_error( "Unable to create HDF5 space: " );
  if( (dataset = H5Dcreate2(gid, c_str, dtype, sid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT)) < 0)
    throw runtime_error( "Unable to create HDF5 dataset " );

  delete[] c_str;

    /* set up dimensions of the slab this process accesses */

  start[0] = commRank*Totnumel/commRanks;

  hypercount[0] = numel;
  stride[0] = 1;

  file_dataspace = H5Dget_space (dataset);
   
  ret=H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, start, stride,
			  hypercount, NULL);

  sizedims[0] = numel;

    mem_dataspace = H5Screate_simple (1, sizedims, NULL);

    // write data independently
    ret = H5Dwrite(dataset, dtype, mem_dataspace, file_dataspace,
		   H5P_DEFAULT, (void *)buf);

    // release dataspaceID
    H5Sclose (file_dataspace);
    H5Sclose (mem_dataspace);
    H5Sclose (sid);
    ret=H5Dclose(dataset);

    
    //
    // Create the CRC dataset
    //

    std::strcpy(c_str3, D.c_str());
    strcat(c_str3, "_CRC");

    file_dataspace = H5Screate_simple (1, adims, NULL);
    dataset = H5Dcreate2(gid, c_str3, H5T_NATIVE_ULONG, file_dataspace,
			H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    delete[] c_str3;

    start[0] =  commRank;

    ret=H5Sselect_elements(file_dataspace, H5S_SELECT_SET, 1, start);

    sizedims[0] = 1;

    mem_dataspace = H5Screate_simple (1, sizedims, NULL);

    // write data independently
    ret = H5Dwrite(dataset, H5T_NATIVE_ULONG, mem_dataspace, file_dataspace,
		   H5P_DEFAULT, (void *)CRC);

    H5Sclose (file_dataspace);
    H5Sclose (mem_dataspace);
    H5Dclose (dataset);


//    while (count > 0) {
//      MPI_Status status;
//     if (MPI_File_write_at(FH, offset, (void *) buf, count, MPI_BYTE, &status) != MPI_SUCCESS)
//       throw runtime_error("Unable to write " + D + " to file: " + FileName);

//     int scount;
//     (void) MPI_Get_count(&status, MPI_BYTE, &scount);

//     count -= scount;
//     buf = ((char *) buf) + scount;
//     offset += scount;
//    }
}

void GenericFileIO_HDFCollective::read(void *buf, size_t count, off_t offset,
                             const std::string &D) {
  int Continue = 0;
  

//   do {
//     MPI_Status status;
//     if (MPI_File_read_at_all(FH, offset, buf, count, MPI_BYTE, &status) != MPI_SUCCESS)
//       throw runtime_error("Unable to read " + D + " from file: " + FileName);

//     int scount;
//     (void) MPI_Get_count(&status, MPI_BYTE, &scount);

//     count -= scount;
//     buf = ((char *) buf) + scount;
//     offset += scount;

//     int NeedContinue = (count > 0);
//     MPI_Allreduce(&NeedContinue, &Continue, 1, MPI_INT, MPI_SUM, Comm);
//   } while (Continue);
}

void GenericFileIO_HDFCollective::write(const void *buf, size_t count, off_t offset,
                              const std::string &D) {
  int Continue = 0;

  //  cout << "GenericFileIO_HDFCollective" << endl;

//   do {
//     MPI_Status status;
//     if (MPI_File_write_at_all(FH, offset, (void *) buf, count, MPI_BYTE, &status) != MPI_SUCCESS)
//       throw runtime_error("Unable to write " + D + " to file: " + FileName);

//     int scount;
//     (void) MPI_Get_count(&status, MPI_BYTE, &scount);

//     count -= scount;
//     buf = ((char *) buf) + scount;
//     offset += scount;

//     int NeedContinue = (count > 0);
//     MPI_Allreduce(&NeedContinue, &Continue, 1, MPI_INT, MPI_SUM, Comm);
//   } while (Continue);
}
#endif

GenericFileIO_POSIX::~GenericFileIO_POSIX() {
  if (FH != -1) close(FH);
}

void GenericFileIO_POSIX::open(const std::string &FN, bool ForReading) {
  FileName = FN;

  int flags = ForReading ? O_RDONLY : (O_WRONLY | O_CREAT);
  int mode = S_IRUSR | S_IWUSR | S_IRGRP;
  errno = 0;
  if ((FH = ::open(FileName.c_str(), flags, mode)) == -1)
    throw runtime_error((!ForReading ? "Unable to create the file: " :
                                       "Unable to open the file: ") +
                        FileName + ": " + strerror(errno));
}

void GenericFileIO_POSIX::setSize(size_t sz) {
  if (ftruncate(FH, sz) == -1)
    throw runtime_error("Unable to set size for file: " + FileName);
}

void GenericFileIO_POSIX::read(void *buf, size_t count, off_t offset,
                               const std::string &D) {
  while (count > 0) {
    ssize_t scount;
    errno = 0;
    if ((scount = pread(FH, buf, count, offset)) == -1) {
      if (errno == EINTR)
        continue;

      throw runtime_error("Unable to read " + D + " from file: " + FileName +
                          ": " + strerror(errno));
    }

    count -= scount;
    buf = ((char *) buf) + scount;
    offset += scount;
  }
}

void GenericFileIO_POSIX::write(const void *buf, size_t count, off_t offset,
                                const std::string &D) {
  while (count > 0) {
    ssize_t scount;
    errno = 0;
    if ((scount = pwrite(FH, buf, count, offset)) == -1) {
      if (errno == EINTR)
        continue;

      throw runtime_error("Unable to write " + D + " to file: " + FileName +
                          ": " + strerror(errno));
    }

    count -= scount;
    buf = ((char *) buf) + scount;
    offset += scount;
  }
}

static bool isBigEndian() {
  const uint32_t one = 1;
  return !(*((char *)(&one)));
}

static void bswap(void *v, size_t s) {
  char *p = (char *) v;
  for (size_t i = 0; i < s/2; ++i)
    std::swap(p[i], p[s - (i+1)]);
}

// Using #pragma pack here, instead of __attribute__((packed)) because xlc, at
// least as of v12.1, won't take __attribute__((packed)) on non-POD and/or
// templated types.
#pragma pack(1)

template <typename T, bool IsBigEndian>
struct endian_specific_value {
  operator T() const {
    T rvalue = value;
    if (IsBigEndian != isBigEndian())
      bswap(&rvalue, sizeof(T));

    return rvalue;
  };

  endian_specific_value &operator = (T nvalue) {
    if (IsBigEndian != isBigEndian())
      bswap(&nvalue, sizeof(T));

    value = nvalue;
    return *this;
  }

  endian_specific_value &operator += (T nvalue) {
    *this = *this + nvalue;
    return *this;
  }

  endian_specific_value &operator -= (T nvalue) {
    *this = *this - nvalue;
    return *this;
  }

private:
  T value;
};

static const size_t CRCSize = 8;

static const size_t MagicSize = 8;
static const char *MagicBE = "HACC01B";
static const char *MagicLE = "HACC01L";

template <bool IsBigEndian>
struct GlobalHeader {
  char Magic[MagicSize];
  endian_specific_value<uint64_t, IsBigEndian> HeaderSize;
  endian_specific_value<uint64_t, IsBigEndian> NElems; // The global total
  endian_specific_value<uint64_t, IsBigEndian> Dims[3];
  endian_specific_value<uint64_t, IsBigEndian> NVars;
  endian_specific_value<uint64_t, IsBigEndian> VarsSize;
  endian_specific_value<uint64_t, IsBigEndian> VarsStart;
  endian_specific_value<uint64_t, IsBigEndian> NRanks;
  endian_specific_value<uint64_t, IsBigEndian> RanksSize;
  endian_specific_value<uint64_t, IsBigEndian> RanksStart;
  endian_specific_value<uint64_t, IsBigEndian> GlobalHeaderSize;
  endian_specific_value<double,   IsBigEndian> PhysOrigin[3];
  endian_specific_value<double,   IsBigEndian> PhysScale[3];
  endian_specific_value<uint64_t, IsBigEndian> BlocksSize;
  endian_specific_value<uint64_t, IsBigEndian> BlocksStart;
};

enum {
  FloatValue          = (1 << 0),
  SignedValue         = (1 << 1),
  ValueIsPhysCoordX   = (1 << 2),
  ValueIsPhysCoordY   = (1 << 3),
  ValueIsPhysCoordZ   = (1 << 4),
  ValueMaybePhysGhost = (1 << 5)
};

static const size_t NameSize = 256;
template <bool IsBigEndian>
struct VariableHeader {
  char Name[NameSize];
  endian_specific_value<uint64_t, IsBigEndian> Flags;
  endian_specific_value<uint64_t, IsBigEndian> Size;
  endian_specific_value<uint64_t, IsBigEndian> ElementSize;
};

template <bool IsBigEndian>
struct RankHeader {
  endian_specific_value<uint64_t, IsBigEndian> Coords[3];
  endian_specific_value<uint64_t, IsBigEndian> NElems;
  endian_specific_value<uint64_t, IsBigEndian> Start;
  endian_specific_value<uint64_t, IsBigEndian> GlobalRank;
};

static const size_t FilterNameSize = 8;
static const size_t MaxFilters = 4;
template <bool IsBigEndian>
struct BlockHeader {
  char Filters[MaxFilters][FilterNameSize];
  endian_specific_value<uint64_t, IsBigEndian> Start;
  endian_specific_value<uint64_t, IsBigEndian> Size;
};

template <bool IsBigEndian>
struct CompressHeader {
  endian_specific_value<uint64_t, IsBigEndian> OrigCRC;
};
const char *CompressName = "BLOSC";

#pragma pack()

unsigned GenericIO::DefaultFileIOType = FileIOPOSIX;
int GenericIO::DefaultPartition = 0;
bool GenericIO::DefaultShouldCompress = false;

#ifndef GENERICIO_NO_MPI
std::size_t GenericIO::CollectiveMPIIOThreshold = 0;
#endif

static bool blosc_initialized = false;

#ifndef GENERICIO_NO_MPI
void GenericIO::write() {
  if (isBigEndian())
    write<true>();
  else
    write<false>();
}

// Note: writing errors are not currently recoverable (one rank may fail
// while the others don't).
template <bool IsBigEndian>
void GenericIO::write() {
  const char *Magic = IsBigEndian ? MagicBE : MagicLE;
#ifdef GENERICIO_HAVE_HDF
  hid_t fid, space, dset, attr, filetype, atype, gid, aid, tid1, rid1, rid2, dxpl_id;
  hid_t fapl_id;  // File access templates
  hid_t sid, *dataset, *dataset_CRC;
  hsize_t dims[1]; // dataspace dim sizes
  hsize_t adim[1];
  hsize_t start[1];
  hsize_t start_CRC[1];			/* for hyperslab setting */
  hsize_t hypercount[1], stride[1];	/* for hyperslab setting */
  hsize_t sizedims[1], adims[1];

  hid_t file_dataspace, mem_dataspace;
  hid_t file_dataspace_CRC, mem_dataspace_CRC;
  /* Groups */
  hid_t  grp_l_id;
  hid_t  grp_p_id;
  hid_t  grp_s_id;
  herr_t ret;
#endif
  double timestep[1];
  char notes[] = {"Important notes go here"};

  size_t *token_size;
  size_t *token_size_CRC;
  void **dset_token;
  void **dset_token_CRC;

  uint64_t FileSize = 0;

  int NRanks, Rank;
  MPI_Comm_rank(Comm, &Rank);
  MPI_Comm_size(Comm, &NRanks);

#ifdef __bgq__
  MPI_Barrier(Comm);
#endif
  MPI_Comm_split(Comm, Partition, Rank, &SplitComm);

  int SplitNRanks, SplitRank;
  MPI_Comm_rank(SplitComm, &SplitRank);
  MPI_Comm_size(SplitComm, &SplitNRanks);

  string LocalFileName;
  if (SplitNRanks != NRanks) {
    if (Rank == 0) {
      // In split mode, the specified file becomes the rank map, and the real
      // data is partitioned.

      vector<int> MapRank, MapPartition;
      MapRank.resize(NRanks);
      for (int i = 0; i < NRanks; ++i) MapRank[i] = i;

      MapPartition.resize(NRanks);
      MPI_Gather(&Partition, 1, MPI_INT, &MapPartition[0], 1, MPI_INT, 0, Comm);

      GenericIO GIO(MPI_COMM_SELF, FileName, FileIOType);
      GIO.setNumElems(NRanks);
      GIO.addVariable("$rank", MapRank); /* this is for use by humans; the reading
                                            code assumes that the partitions are in
                                            rank order */
      GIO.addVariable("$partition", MapPartition);

      vector<int> CX, CY, CZ;
      int TopoStatus;
      MPI_Topo_test(Comm, &TopoStatus);
      if (TopoStatus == MPI_CART) {
        CX.resize(NRanks);
        CY.resize(NRanks);
        CZ.resize(NRanks);

        for (int i = 0; i < NRanks; ++i) {
          int C[3];
          MPI_Cart_coords(Comm, i, 3, C);

          CX[i] = C[0];
          CY[i] = C[1];
          CZ[i] = C[2];
        }

        GIO.addVariable("$x", CX);
        GIO.addVariable("$y", CY);
        GIO.addVariable("$z", CZ);
      }

      GIO.write();
    } else {
      MPI_Gather(&Partition, 1, MPI_INT, 0, 0, MPI_INT, 0, Comm);
    }

    stringstream ss;
    ss << FileName << "#" << Partition;
    LocalFileName = ss.str();
  } else {
    LocalFileName = FileName;
  }

  RankHeader<IsBigEndian> RHLocal;
  int Dims[3], Periods[3], Coords[3];

  int TopoStatus;
  MPI_Topo_test(Comm, &TopoStatus);
  if (TopoStatus == MPI_CART) {
    MPI_Cart_get(Comm, 3, Dims, Periods, Coords);
  } else {
    Dims[0] = NRanks;
    std::fill(Dims + 1, Dims + 3, 1);
    std::fill(Periods, Periods + 3, 0);
    Coords[0] = Rank;
    std::fill(Coords + 1, Coords + 3, 0);
  }

  std::copy(Coords, Coords + 3, RHLocal.Coords);
  RHLocal.NElems = NElems;
  RHLocal.Start = 0;
  RHLocal.GlobalRank = Rank;

  bool ShouldCompress = DefaultShouldCompress;
  const char *EnvStr = getenv("GENERICIO_COMPRESS");
  if (EnvStr) {
    int Mod = atoi(EnvStr);
    ShouldCompress = (Mod > 0);
  }

  bool NeedsBlockHeaders = ShouldCompress;
  EnvStr = getenv("GENERICIO_FORCE_BLOCKS");
  if (!NeedsBlockHeaders && EnvStr) {
    int Mod = atoi(EnvStr);
    NeedsBlockHeaders = (Mod > 0);
  }

  vector<BlockHeader<IsBigEndian> > LocalBlockHeaders;
  vector<void *> LocalData;
  vector<bool> LocalHasExtraSpace;
  vector<vector<unsigned char> > LocalCData;
  if (NeedsBlockHeaders) {
    LocalBlockHeaders.resize(Vars.size());
    LocalData.resize(Vars.size());
    LocalHasExtraSpace.resize(Vars.size());
    if (ShouldCompress)
      LocalCData.resize(Vars.size());

    for (size_t i = 0; i < Vars.size(); ++i) {
      // Filters null by default, leave null starting address (needs to be
      // calculated by the header-writing rank).
      memset(&LocalBlockHeaders[i], 0, sizeof(BlockHeader<IsBigEndian>));
      if (ShouldCompress) {
        LocalCData[i].resize(sizeof(CompressHeader<IsBigEndian>));

        CompressHeader<IsBigEndian> *CH = (CompressHeader<IsBigEndian>*) &LocalCData[i][0];
        CH->OrigCRC = crc64_omp(Vars[i].Data, Vars[i].Size*NElems);

#ifdef _OPENMP
#pragma omp master
  {
#endif

       if (!blosc_initialized) {
         blosc_init();
         blosc_initialized = true;
       }

#ifdef _OPENMP
       blosc_set_nthreads(omp_get_max_threads());
  }
#endif

        LocalCData[i].resize(LocalCData[i].size() + NElems*Vars[i].Size);
        if (blosc_compress(9, 1, Vars[i].Size, NElems*Vars[i].Size, Vars[i].Data,
                           &LocalCData[i][0] + sizeof(CompressHeader<IsBigEndian>),
                           NElems*Vars[i].Size) <= 0)
          goto nocomp;

        strncpy(LocalBlockHeaders[i].Filters[0], CompressName, FilterNameSize);
        size_t CNBytes, CCBytes, CBlockSize;
        blosc_cbuffer_sizes(&LocalCData[i][0] + sizeof(CompressHeader<IsBigEndian>),
                            &CNBytes, &CCBytes, &CBlockSize);
        LocalCData[i].resize(CCBytes + sizeof(CompressHeader<IsBigEndian>));

        LocalBlockHeaders[i].Size = LocalCData[i].size();
        LocalCData[i].resize(LocalCData[i].size() + CRCSize);
        LocalData[i] = &LocalCData[i][0];
        LocalHasExtraSpace[i] = true;
      } else {
nocomp:
        LocalBlockHeaders[i].Size = NElems*Vars[i].Size;
        LocalData[i] = Vars[i].Data;
        LocalHasExtraSpace[i] = Vars[i].HasExtraSpace;
      }
    }
  }

  double StartTime = MPI_Wtime();

  if (SplitRank == 0) {
    uint64_t HeaderSize = sizeof(GlobalHeader<IsBigEndian>) + Vars.size()*sizeof(VariableHeader<IsBigEndian>) +
                          SplitNRanks*sizeof(RankHeader<IsBigEndian>) + CRCSize;
    if (NeedsBlockHeaders)
      HeaderSize += SplitNRanks*Vars.size()*sizeof(BlockHeader<IsBigEndian>);

    vector<char> Header(HeaderSize, 0);
    GlobalHeader<IsBigEndian> *GH = (GlobalHeader<IsBigEndian> *) &Header[0];
    std::copy(Magic, Magic + MagicSize, GH->Magic);
    GH->HeaderSize = HeaderSize - CRCSize;
    GH->NElems = NElems; // This will be updated later
    std::copy(Dims, Dims + 3, GH->Dims);
    GH->NVars = Vars.size();
    GH->VarsSize = sizeof(VariableHeader<IsBigEndian>);
    GH->VarsStart = sizeof(GlobalHeader<IsBigEndian>);
    GH->NRanks = SplitNRanks;
    GH->RanksSize = sizeof(RankHeader<IsBigEndian>);
    GH->RanksStart = GH->VarsStart + Vars.size()*sizeof(VariableHeader<IsBigEndian>);
    GH->GlobalHeaderSize = sizeof(GlobalHeader<IsBigEndian>);
    std::copy(PhysOrigin, PhysOrigin + 3, GH->PhysOrigin);
    std::copy(PhysScale,  PhysScale  + 3, GH->PhysScale);
    if (!NeedsBlockHeaders) {
      GH->BlocksSize = GH->BlocksStart = 0;
    } else {
      GH->BlocksSize = sizeof(BlockHeader<IsBigEndian>);
      GH->BlocksStart = GH->RanksStart + SplitNRanks*sizeof(RankHeader<IsBigEndian>);
    }

    uint64_t RecordSize = 0;
    VariableHeader<IsBigEndian> *VH = (VariableHeader<IsBigEndian> *) &Header[GH->VarsStart];
    for (size_t i = 0; i < Vars.size(); ++i, ++VH) {
      string VName(Vars[i].Name);
      VName.resize(NameSize);

      std::copy(VName.begin(), VName.end(), VH->Name);
      uint64_t VFlags = 0;
      if (Vars[i].IsFloat)  VFlags |= FloatValue;
      if (Vars[i].IsSigned) VFlags |= SignedValue;
      if (Vars[i].IsPhysCoordX) VFlags |= ValueIsPhysCoordX;
      if (Vars[i].IsPhysCoordY) VFlags |= ValueIsPhysCoordY;
      if (Vars[i].IsPhysCoordZ) VFlags |= ValueIsPhysCoordZ;
      if (Vars[i].MaybePhysGhost) VFlags |= ValueMaybePhysGhost;
      VH->Flags = VFlags;
      RecordSize += VH->Size = Vars[i].Size;
      VH->ElementSize = Vars[i].ElementSize;
    }

    MPI_Gather(&RHLocal, sizeof(RHLocal), MPI_BYTE,
               &Header[GH->RanksStart], sizeof(RHLocal),
               MPI_BYTE, 0, SplitComm);

    if (NeedsBlockHeaders) {
      MPI_Gather(&LocalBlockHeaders[0],
                 Vars.size()*sizeof(BlockHeader<IsBigEndian>), MPI_BYTE,
                 &Header[GH->BlocksStart],
                 Vars.size()*sizeof(BlockHeader<IsBigEndian>), MPI_BYTE,
                 0, SplitComm);

      BlockHeader<IsBigEndian> *BH = (BlockHeader<IsBigEndian> *) &Header[GH->BlocksStart];
      for (int i = 0; i < SplitNRanks; ++i)
      for (size_t j = 0; j < Vars.size(); ++j, ++BH) {
        if (i == 0 && j == 0)
          BH->Start = HeaderSize;
        else
          BH->Start = BH[-1].Start + BH[-1].Size + CRCSize;
      }

      RankHeader<IsBigEndian> *RH = (RankHeader<IsBigEndian> *) &Header[GH->RanksStart];
      RH->Start = HeaderSize; ++RH;
      for (int i = 1; i < SplitNRanks; ++i, ++RH) {
        RH->Start =
          ((BlockHeader<IsBigEndian> *) &Header[GH->BlocksStart])[i*Vars.size()].Start;
        GH->NElems += RH->NElems;
      }

      // Compute the total file size.
      uint64_t LastData = BH[-1].Size + CRCSize;
      FileSize = BH[-1].Start + LastData;
    } else {
      RankHeader<IsBigEndian> *RH = (RankHeader<IsBigEndian> *) &Header[GH->RanksStart];
      RH->Start = HeaderSize; ++RH;
      for (int i = 1; i < SplitNRanks; ++i, ++RH) {
        uint64_t PrevNElems = RH[-1].NElems;
        uint64_t PrevData = PrevNElems*RecordSize + CRCSize*Vars.size();
        RH->Start = RH[-1].Start + PrevData;
        GH->NElems += RH->NElems;
      }

      // Compute the total file size.
      uint64_t LastNElems = RH[-1].NElems;
      uint64_t LastData = LastNElems*RecordSize + CRCSize*Vars.size();
      FileSize = RH[-1].Start + LastData;
    }

    // Now that the starting offset has been computed, send it back to each rank.
    MPI_Scatter(&Header[GH->RanksStart], sizeof(RHLocal),
                MPI_BYTE, &RHLocal, sizeof(RHLocal),
                MPI_BYTE, 0, SplitComm);

    if (NeedsBlockHeaders)
      MPI_Scatter(&Header[GH->BlocksStart],
                  sizeof(BlockHeader<IsBigEndian>)*Vars.size(), MPI_BYTE,
                  &LocalBlockHeaders[0],
                  sizeof(BlockHeader<IsBigEndian>)*Vars.size(), MPI_BYTE,
                  0, SplitComm);

    uint64_t HeaderCRC = crc64_omp(&Header[0], HeaderSize - CRCSize);
    crc64_invert(HeaderCRC, &Header[HeaderSize - CRCSize]);
#ifdef GENERICIO_HAVE_HDF
    if (FileIOType == FileIOHDF)
      FH.get() = new GenericFileIO_HDF(MPI_COMM_SELF);
    else
#endif 
    if (FileIOType == FileIOMPI)
      FH.get() = new GenericFileIO_MPI(MPI_COMM_SELF);
    else if (FileIOType == FileIOMPICollective)
      FH.get() = new GenericFileIO_MPICollective(MPI_COMM_SELF);
    else
      FH.get() = new GenericFileIO_POSIX();

    FH.get()->open(LocalFileName);
    FH.get()->setSize(FileSize);
    FH.get()->write(&Header[0], HeaderSize, 0, "header");

    close();
  } else {
    MPI_Gather(&RHLocal, sizeof(RHLocal), MPI_BYTE, 0, 0, MPI_BYTE, 0, SplitComm);
    if (NeedsBlockHeaders)
      MPI_Gather(&LocalBlockHeaders[0], Vars.size()*sizeof(BlockHeader<IsBigEndian>),
                 MPI_BYTE, 0, 0, MPI_BYTE, 0, SplitComm);
    MPI_Scatter(0, 0, MPI_BYTE, &RHLocal, sizeof(RHLocal), MPI_BYTE, 0, SplitComm);
    if (NeedsBlockHeaders)
      MPI_Scatter(0, 0, MPI_BYTE, &LocalBlockHeaders[0], sizeof(BlockHeader<IsBigEndian>)*Vars.size(),
                  MPI_BYTE, 0, SplitComm);
  }

  MPI_Barrier(SplitComm);
#ifdef GENERICIO_HAVE_HDF
  if (FileIOType == FileIOHDF)
    FH.get() = new GenericFileIO_HDF(SplitComm);
  else
#endif
  if (FileIOType == FileIOMPI)
    FH.get() = new GenericFileIO_MPI(SplitComm);
  else if (FileIOType == FileIOMPICollective)
    FH.get() = new GenericFileIO_MPICollective(SplitComm);
  else
    FH.get() = new GenericFileIO_POSIX();

  FH.get()->open(LocalFileName);

  uint64_t Offset = RHLocal.Start;
#ifdef GENERICIO_HAVE_HDF
  GenericFileIO_HDF *gfio_hdf;

  if (FileIOType == FileIOHDF) {
    gfio_hdf = dynamic_cast<GenericFileIO_HDF *> (FH.get());
    fid = gfio_hdf->get_fileid();

    filetype = H5Tcopy (H5T_C_S1);
    ret = H5Tset_size (filetype, H5T_VARIABLE);

    // Create a header dataset containing file's metadata information

    gid   = H5Gcreate2(fid, "Variables", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    //
    // Create dataset with a null dataspace.
    //
    space = H5Screate (H5S_NULL);
    
    // Create dataspace.  Setting maximum size to NULL sets the maximum
    // size to be the current size.
    //
    adim[0] = 1;
    aid = H5Screate(H5S_SIMPLE);
    ret = H5Sset_extent_simple(aid, 1, adim, NULL);
    //
    // Create the attribute 
    //
    timestep[0] = 123.123; 
    attr = H5Acreate (gid, "Time step", H5T_NATIVE_DOUBLE, aid, H5P_DEFAULT, H5P_DEFAULT);
    ret  = H5Awrite (attr, H5T_NATIVE_DOUBLE, timestep);
    ret = H5Aclose (attr);
    
    ret = H5Sclose (aid);
    
    aid = H5Screate(H5S_SCALAR);
    atype = H5Tcopy(H5T_C_S1);
    H5Tset_size(atype, 23);
    H5Tset_strpad(atype,H5T_STR_NULLTERM);
    
    attr = H5Acreate (gid, "Notes", atype, aid, H5P_DEFAULT, H5P_DEFAULT);
    ret  = H5Awrite (attr, atype, notes);
    ret  = H5Aclose (attr);
    ret = H5Sclose (aid);
    ret = H5Tclose (filetype);
    ret = H5Gclose (gid);
    
    gid = H5Gopen2(fid, "Variables", H5P_DEFAULT);
  }
#endif
  for (size_t i = 0; i < Vars.size(); ++i) {
    uint64_t WriteSize = NeedsBlockHeaders ?
                         LocalBlockHeaders[i].Size : NElems*Vars[i].Size;
    void *Data = NeedsBlockHeaders ? LocalData[i] : Vars[i].Data;
    uint64_t CRC = crc64_omp(Data, WriteSize);

    bool HasExtraSpace = NeedsBlockHeaders ?
                         LocalHasExtraSpace[i] : Vars[i].HasExtraSpace;
    char *CRCLoc = HasExtraSpace ?  ((char *) Data) + WriteSize : (char *) &CRC;

    if (NeedsBlockHeaders)
      Offset = LocalBlockHeaders[i].Start;

    // When using extra space for the CRC write, preserve the original contents.
    char CRCSave[CRCSize];
    if (HasExtraSpace)
      std::copy(CRCLoc, CRCLoc + CRCSize, CRCSave);


    if (HasExtraSpace) {
#ifdef GENERICIO_HAVE_HDF
      if (FileIOType == FileIOHDF) {
	hid_t dtype;
	CRC = crc64_omp(Data, NElems);
	if( Vars[i].Name.compare("id") == 0) {
	  dtype = H5T_NATIVE_LONG;
	} else if( Vars[i].Name.compare("mask") == 0) {
	  dtype = H5T_NATIVE_B16;
	}else {
	  dtype = H5T_NATIVE_FLOAT;
	}
	gfio_hdf->write_hdf(Data, WriteSize + CRCSize, Offset, Vars[i].Name, dtype, NElems, &CRC, gid, TotElem);
      } else 
#endif
	crc64_invert(CRC, CRCLoc);
	FH.get()->write(Data, WriteSize + CRCSize, Offset, Vars[i].Name + " with CRC");
    } else {
      crc64_invert(CRC, CRCLoc);
      FH.get()->write(Data, WriteSize, Offset, Vars[i].Name);
      FH.get()->write(CRCLoc, CRCSize, Offset + WriteSize, Vars[i].Name + " CRC");
    }

    if (HasExtraSpace)
       std::copy(CRCSave, CRCSave + CRCSize, CRCLoc);

    Offset += WriteSize + CRCSize;
  }
#ifdef GENERICIO_HAVE_HDF
  if (FileIOType == FileIOHDF)
    ret = H5Gclose(gid);
#endif
  close();
  MPI_Barrier(Comm);

  double EndTime = MPI_Wtime();
  double TotalTime = EndTime - StartTime;
  double MaxTotalTime;
  MPI_Reduce(&TotalTime, &MaxTotalTime, 1, MPI_DOUBLE, MPI_MAX, 0, Comm);

  if (SplitNRanks != NRanks) {
    uint64_t ContribFileSize = (SplitRank == 0) ? FileSize : 0;
    MPI_Reduce(&ContribFileSize, &FileSize, 1, MPI_UINT64_T, MPI_SUM, 0, Comm);
  }

  if (Rank == 0) {
    double Rate = ((double) FileSize) / MaxTotalTime / (1024.*1024.);
    cout << "Wrote " << Vars.size() << " variables to " << FileName <<
            " (" << FileSize << " bytes) in " << MaxTotalTime << "s: " <<
            Rate << " MB/s" << endl;
  }

  MPI_Comm_free(&SplitComm);
  SplitComm = MPI_COMM_NULL;
}
#endif // GENERICIO_NO_MPI

template <bool IsBigEndian>
void GenericIO::readHeaderLeader(void *GHPtr, MismatchBehavior MB, int NRanks,
                                 int Rank, int SplitNRanks,
                                 string &LocalFileName, uint64_t &HeaderSize,
                                 vector<char> &Header) {
  GlobalHeader<IsBigEndian> &GH = *(GlobalHeader<IsBigEndian> *) GHPtr;

  if (MB == MismatchDisallowed) {
    if (SplitNRanks != (int) GH.NRanks) {
      stringstream ss;
      ss << "Won't read " << LocalFileName << ": communicator-size mismatch: " <<
            "current: " << SplitNRanks << ", file: " << GH.NRanks;
      throw runtime_error(ss.str());
    }

#ifndef GENERICIO_NO_MPI
    int TopoStatus;
    MPI_Topo_test(Comm, &TopoStatus);
    if (TopoStatus == MPI_CART) {
      int Dims[3], Periods[3], Coords[3];
      MPI_Cart_get(Comm, 3, Dims, Periods, Coords);

      bool DimsMatch = true;
      for (int i = 0; i < 3; ++i) {
        if ((uint64_t) Dims[i] != GH.Dims[i]) {
          DimsMatch = false;
          break;
        }
      }

      if (!DimsMatch) {
        stringstream ss;
        ss << "Won't read " << LocalFileName <<
              ": communicator-decomposition mismatch: " <<
              "current: " << Dims[0] << "x" << Dims[1] << "x" << Dims[2] <<
              ", file: " << GH.Dims[0] << "x" << GH.Dims[1] << "x" <<
              GH.Dims[2];
        throw runtime_error(ss.str());
      }
    }
#endif
  } else if (MB == MismatchRedistribute && !Redistributing) {
    Redistributing = true;

    int NFileRanks = RankMap.empty() ? (int) GH.NRanks : (int) RankMap.size();
    int NFileRanksPerRank = NFileRanks/NRanks;
    int NRemFileRank = NFileRanks % NRanks;

    if (!NFileRanksPerRank) {
      // We have only the remainder, so the last NRemFileRank ranks get one
      // file rank, and the others don't.
      if (NRemFileRank && NRanks - Rank <= NRemFileRank)
        SourceRanks.push_back(NRanks - (Rank + 1));
    } else {
      // Since NRemFileRank < NRanks, and we don't want to put any extra memory
      // load on rank 0 (because rank 0's memory load is normally higher than
      // the other ranks anyway), the last NRemFileRank will each take
      // (NFileRanksPerRank+1) file ranks.

      int FirstFileRank = 0, LastFileRank = NFileRanksPerRank - 1;
      for (int i = 1; i <= Rank; ++i) {
        FirstFileRank = LastFileRank + 1;
        LastFileRank  = FirstFileRank + NFileRanksPerRank - 1;

        if (NRemFileRank && NRanks - i <= NRemFileRank)
          ++LastFileRank;
      }

      for (int i = FirstFileRank; i <= LastFileRank; ++i)
        SourceRanks.push_back(i);
    }
  }

  HeaderSize = GH.HeaderSize;
  Header.resize(HeaderSize + CRCSize, 0xFE /* poison */);
  FH.get()->read(&Header[0], HeaderSize + CRCSize, 0, "header");

  uint64_t CRC = crc64_omp(&Header[0], HeaderSize + CRCSize);
  if (CRC != (uint64_t) -1) {
    throw runtime_error("Header CRC check failed: " + LocalFileName);
  }
}

  void GenericIO::openAndReadHeader_HDF(size_t *Numel, bool MustMatch, int EffRank, bool CheckPartMap) {
  int NRanks, Rank;
#ifndef GENERICIO_NO_MPI
  MPI_Comm_rank(Comm, &Rank);
  MPI_Comm_size(Comm, &NRanks);
#else
  Rank = 0;
  NRanks = 1;
#endif

  cout << "openAndReadHeader_HDF" << endl;

#ifndef GENERICIO_NO_MPI
  GenericIO GIO(MPI_COMM_SELF, FileName, FileIOType);
#ifdef GENERICIO_HAVE_HDF 
  FH.get() = new GenericFileIO_HDF(MPI_COMM_SELF);
#endif
#else
  GenericIO GIO(FileName, FileIOType);
#endif
  FH.get()->open(FileName, true);

#ifdef GENERICIO_HAVE_HDF
  GenericFileIO_HDF *gfio_hdf = dynamic_cast<GenericFileIO_HDF *> (FH.get());
  *Numel = gfio_hdf->get_NumElem();
#endif


  return;
}

// Note: Errors from this function should be recoverable. This means that if
// one rank throws an exception, then all ranks should.
void GenericIO::openAndReadHeader(MismatchBehavior MB, int EffRank, bool CheckPartMap) {
  int NRanks, Rank;
#ifndef GENERICIO_NO_MPI
  MPI_Comm_rank(Comm, &Rank);
  MPI_Comm_size(Comm, &NRanks);
#else
  Rank = 0;
  NRanks = 1;
#endif

  if (EffRank == -1)
    EffRank = MB == MismatchRedistribute ? 0 : Rank;

  if (RankMap.empty() && CheckPartMap) {
    // First, check to see if the file is a rank map.
    unsigned long RanksInMap = 0;
    if (Rank == 0) {
      try {
#ifndef GENERICIO_NO_MPI
        GenericIO GIO(MPI_COMM_SELF, FileName, FileIOType);
#ifdef GENERICIO_HAVE_HDF
        FH.get() = new GenericFileIO_HDF(MPI_COMM_SELF);
#endif
#else
        GenericIO GIO(FileName, FileIOType);
#endif
        GIO.openAndReadHeader(MismatchDisallowed, 0, false);
        RanksInMap = GIO.readNumElems();

        RankMap.resize(RanksInMap + GIO.requestedExtraSpace()/sizeof(int));
        GIO.addVariable("$partition", RankMap, true);
#ifdef GENERICIO_HAVE_HDF
	if (FileIOType == FileIOHDF)
	  GIO.readDataHDF(0, false);
	else
#endif
	  GIO.readData(0, false);
        RankMap.resize(RanksInMap);
      } catch (...) {
        RankMap.clear();
        RanksInMap = 0;
      }
    }

#ifndef GENERICIO_NO_MPI
    MPI_Bcast(&RanksInMap, 1, MPI_UNSIGNED_LONG, 0, Comm);
    if (RanksInMap > 0) {
      RankMap.resize(RanksInMap);
      MPI_Bcast(&RankMap[0], RanksInMap, MPI_INT, 0, Comm);
    }
#endif
  }

#ifndef GENERICIO_NO_MPI
  if (SplitComm != MPI_COMM_NULL)
    MPI_Comm_free(&SplitComm);
#endif

  string LocalFileName;
  if (RankMap.empty()) {
    LocalFileName = FileName;
#ifndef GENERICIO_NO_MPI
    MPI_Comm_dup(MB == MismatchRedistribute ? MPI_COMM_SELF : Comm, &SplitComm);
#endif
  } else {
    stringstream ss;
    ss << FileName << "#" << RankMap[EffRank];
    LocalFileName = ss.str();
#ifndef GENERICIO_NO_MPI
    if (MB == MismatchRedistribute) {
      MPI_Comm_dup(MPI_COMM_SELF, &SplitComm);
    } else {
#ifdef __bgq__
      MPI_Barrier(Comm);
#endif
      MPI_Comm_split(Comm, RankMap[EffRank], Rank, &SplitComm);
    }
#endif
  }

  if (LocalFileName == OpenFileName)
    return;
  FH.close();

  int SplitNRanks, SplitRank;
#ifndef GENERICIO_NO_MPI
  MPI_Comm_rank(SplitComm, &SplitRank);
  MPI_Comm_size(SplitComm, &SplitNRanks);
#else
  SplitRank = 0;
  SplitNRanks = 1;
#endif

  uint64_t HeaderSize;
  vector<char> Header;

  if (SplitRank == 0) {
#ifdef GENERICIO_HAVE_MPI
    if (FileIOType == FileIOHDF)
       FH.get() = new GenericFileIO_HDF(MPI_COMM_SELF);
    else if (FileIOType == FileIOMPI)
      FH.get() = new GenericFileIO_MPI(MPI_COMM_SELF);
    else if (FileIOType == FileIOMPICollective)
      FH.get() = new GenericFileIO_MPICollective(MPI_COMM_SELF);
    else
#endif
      FH.get() = new GenericFileIO_POSIX();

#ifndef GENERICIO_NO_MPI
    char True = 1, False = 0;
#endif
    if (FileIOType != FileIOHDF) {
    try {
      FH.get()->open(LocalFileName, true);

      GlobalHeader<false> GH; // endianness does not matter yet...
      FH.get()->read(&GH, sizeof(GlobalHeader<false>), 0, "global header");

      if (string(GH.Magic, GH.Magic + MagicSize - 1) == MagicLE) {
        readHeaderLeader<false>(&GH, MB, NRanks, Rank, SplitNRanks, LocalFileName,
                                HeaderSize, Header);
      } else if (string(GH.Magic, GH.Magic + MagicSize - 1) == MagicBE) {
        readHeaderLeader<true>(&GH, MB, NRanks, Rank, SplitNRanks, LocalFileName,
                               HeaderSize, Header);
      } else {
        string Error = "invalid file-type identifier";
        throw runtime_error("Won't read " + LocalFileName + ": " + Error);
      }

#ifndef GENERICIO_NO_MPI
      close();
      MPI_Bcast(&True, 1, MPI_BYTE, 0, SplitComm);
#endif
    } catch (...) {
#ifndef GENERICIO_NO_MPI
      MPI_Bcast(&False, 1, MPI_BYTE, 0, SplitComm);
#endif
      close();
      throw;
    }
  }
  } else {
#ifndef GENERICIO_NO_MPI
    char Okay;
    MPI_Bcast(&Okay, 1, MPI_BYTE, 0, SplitComm);
    if (!Okay)
      throw runtime_error("Failure broadcast from rank 0");
#endif
  }

#ifndef GENERICIO_NO_MPI
  MPI_Bcast(&HeaderSize, 1, MPI_UINT64_T, 0, SplitComm);
#endif

  Header.resize(HeaderSize, 0xFD /* poison */);
#ifndef GENERICIO_NO_MPI
  MPI_Bcast(&Header[0], HeaderSize, MPI_BYTE, 0, SplitComm);
#endif

  FH.getHeaderCache().clear();

  GlobalHeader<false> *GH = (GlobalHeader<false> *) &Header[0];
  FH.setIsBigEndian(string(GH->Magic, GH->Magic + MagicSize - 1) == MagicBE);

  FH.getHeaderCache().swap(Header);
  OpenFileName = LocalFileName;

#ifndef GENERICIO_NO_MPI
  if (!DisableCollErrChecking)
    MPI_Barrier(Comm);

  if (FileIOType == FileIOMPI)
    FH.get() = new GenericFileIO_MPI(SplitComm);
  else if (FileIOType == FileIOMPICollective)
    FH.get() = new GenericFileIO_MPICollective(SplitComm);
  else
    FH.get() = new GenericFileIO_POSIX();

  int OpenErr = 0, TotOpenErr;
  try {
    FH.get()->open(LocalFileName, true);
    MPI_Allreduce(&OpenErr, &TotOpenErr, 1, MPI_INT, MPI_SUM,
                  DisableCollErrChecking ? MPI_COMM_SELF : Comm);
  } catch (...) {
    OpenErr = 1;
    MPI_Allreduce(&OpenErr, &TotOpenErr, 1, MPI_INT, MPI_SUM,
                  DisableCollErrChecking ? MPI_COMM_SELF : Comm);
    throw;
  }

  if (TotOpenErr > 0) {
    stringstream ss;
    ss << TotOpenErr << " ranks failed to open file: " << LocalFileName;
    throw runtime_error(ss.str());
  }
#endif
}

int GenericIO::readNRanks() {
  if (FH.isBigEndian())
    return readNRanks<true>();
  return readNRanks<false>();
}

template <bool IsBigEndian>
int GenericIO::readNRanks() {
  if (RankMap.size())
    return RankMap.size();

  assert(FH.getHeaderCache().size() && "HeaderCache must not be empty");
  GlobalHeader<IsBigEndian> *GH = (GlobalHeader<IsBigEndian> *) &FH.getHeaderCache()[0];
  return (int) GH->NRanks;
}

void GenericIO::readDims(int Dims[3]) {
  if (FH.isBigEndian())
    readDims<true>(Dims);
  else
    readDims<false>(Dims);
}

template <bool IsBigEndian>
void GenericIO::readDims(int Dims[3]) {
  assert(FH.getHeaderCache().size() && "HeaderCache must not be empty");
  GlobalHeader<IsBigEndian> *GH = (GlobalHeader<IsBigEndian> *) &FH.getHeaderCache()[0];
  std::copy(GH->Dims, GH->Dims + 3, Dims);
}

uint64_t GenericIO::readTotalNumElems() {
  if (FH.isBigEndian())
    return readTotalNumElems<true>();
  return readTotalNumElems<false>();
}

template <bool IsBigEndian>
uint64_t GenericIO::readTotalNumElems() {
  if (RankMap.size())
    return (uint64_t) -1;

  assert(FH.getHeaderCache().size() && "HeaderCache must not be empty");
  GlobalHeader<IsBigEndian> *GH = (GlobalHeader<IsBigEndian> *) &FH.getHeaderCache()[0];
  return GH->NElems;
}

void GenericIO::readPhysOrigin(double Origin[3]) {
  if (FH.isBigEndian())
    readPhysOrigin<true>(Origin);
  else
    readPhysOrigin<false>(Origin);
}

// Define a "safe" version of offsetof (offsetof itself might not work for
// non-POD types, and at least xlC v12.1 will complain about this if you try).
#define offsetof_safe(S, F) (size_t(&(S)->F) - size_t(S))

template <bool IsBigEndian>
void GenericIO::readPhysOrigin(double Origin[3]) {
  assert(FH.getHeaderCache().size() && "HeaderCache must not be empty");
  GlobalHeader<IsBigEndian> *GH = (GlobalHeader<IsBigEndian> *) &FH.getHeaderCache()[0];
  if (offsetof_safe(GH, PhysOrigin) >= GH->GlobalHeaderSize) {
    std::fill(Origin, Origin + 3, 0.0);
    return;
  }

  std::copy(GH->PhysOrigin, GH->PhysOrigin + 3, Origin);
}

void GenericIO::readPhysScale(double Scale[3]) {
  if (FH.isBigEndian())
    readPhysScale<true>(Scale);
  else
    readPhysScale<false>(Scale);
}

template <bool IsBigEndian>
void GenericIO::readPhysScale(double Scale[3]) {
  assert(FH.getHeaderCache().size() && "HeaderCache must not be empty");
  GlobalHeader<IsBigEndian> *GH = (GlobalHeader<IsBigEndian> *) &FH.getHeaderCache()[0];
  if (offsetof_safe(GH, PhysScale) >= GH->GlobalHeaderSize) {
    std::fill(Scale, Scale + 3, 0.0);
    return;
  }

  std::copy(GH->PhysScale, GH->PhysScale + 3, Scale);
}

template <bool IsBigEndian>
static size_t getRankIndex(int EffRank, GlobalHeader<IsBigEndian> *GH,
                           vector<int> &RankMap, vector<char> &HeaderCache) {
  if (RankMap.empty())
    return EffRank;

  for (size_t i = 0; i < GH->NRanks; ++i) {
    RankHeader<IsBigEndian> *RH = (RankHeader<IsBigEndian> *) &HeaderCache[GH->RanksStart +
                                                 i*GH->RanksSize];
    if (offsetof_safe(RH, GlobalRank) >= GH->RanksSize)
      return EffRank;

    if ((int) RH->GlobalRank == EffRank)
      return i;
  }

  assert(false && "Index requested of an invalid rank");
  return (size_t) -1;
}

int GenericIO::readGlobalRankNumber(int EffRank) {
  if (FH.isBigEndian())
    return readGlobalRankNumber<true>(EffRank);
  return readGlobalRankNumber<false>(EffRank);
}

template <bool IsBigEndian>
int GenericIO::readGlobalRankNumber(int EffRank) {
  if (EffRank == -1) {
#ifndef GENERICIO_NO_MPI
    MPI_Comm_rank(Comm, &EffRank);
#else
    EffRank = 0;
#endif
  }

  openAndReadHeader(MismatchAllowed, EffRank, false);

  assert(FH.getHeaderCache().size() && "HeaderCache must not be empty");

  GlobalHeader<IsBigEndian> *GH = (GlobalHeader<IsBigEndian> *) &FH.getHeaderCache()[0];
  size_t RankIndex = getRankIndex<IsBigEndian>(EffRank, GH, RankMap, FH.getHeaderCache());

  assert(RankIndex < GH->NRanks && "Invalid rank specified");

  RankHeader<IsBigEndian> *RH = (RankHeader<IsBigEndian> *) &FH.getHeaderCache()[GH->RanksStart +
                                               RankIndex*GH->RanksSize];

  if (offsetof_safe(RH, GlobalRank) >= GH->RanksSize)
    return EffRank;

  return (int) RH->GlobalRank;
}

void GenericIO::getSourceRanks(vector<int> &SR) {
  SR.clear();

  if (Redistributing) {
    std::copy(SourceRanks.begin(), SourceRanks.end(), std::back_inserter(SR));
    return;
  }

  int Rank;
#ifndef GENERICIO_NO_MPI
  MPI_Comm_rank(Comm, &Rank);
#else
  Rank = 0;
#endif

  SR.push_back(Rank);
}

size_t GenericIO::readNumElems(int EffRank) {
  if (EffRank == -1 && Redistributing) {
    DisableCollErrChecking = true;

    size_t TotalSize = 0;
    for (int i = 0, ie = SourceRanks.size(); i != ie; ++i)
      TotalSize += readNumElems(SourceRanks[i]);

    DisableCollErrChecking = false;
    return TotalSize;
  }

  if (FH.isBigEndian())
    return readNumElems<true>(EffRank);
  return readNumElems<false>(EffRank);
}

template <bool IsBigEndian>
size_t GenericIO::readNumElems(int EffRank) {
  if (EffRank == -1) {
#ifndef GENERICIO_NO_MPI
    MPI_Comm_rank(Comm, &EffRank);
#else
    EffRank = 0;
#endif
  }

  openAndReadHeader(Redistributing ? MismatchRedistribute : MismatchAllowed,
                    EffRank, false);

  assert(FH.getHeaderCache().size() && "HeaderCache must not be empty");

  GlobalHeader<IsBigEndian> *GH = (GlobalHeader<IsBigEndian> *) &FH.getHeaderCache()[0];
  size_t RankIndex = getRankIndex<IsBigEndian>(EffRank, GH, RankMap, FH.getHeaderCache());

  assert(RankIndex < GH->NRanks && "Invalid rank specified");

  RankHeader<IsBigEndian> *RH = (RankHeader<IsBigEndian> *) &FH.getHeaderCache()[GH->RanksStart +
                                               RankIndex*GH->RanksSize];
  return (size_t) RH->NElems;
}

void GenericIO::readCoords(int Coords[3], int EffRank) {
  if (EffRank == -1 && Redistributing) {
    std::fill(Coords, Coords + 3, 0);
    return;
  }

  if (FH.isBigEndian())
    readCoords<true>(Coords, EffRank);
  else
    readCoords<false>(Coords, EffRank);
}

template <bool IsBigEndian>
void GenericIO::readCoords(int Coords[3], int EffRank) {
  if (EffRank == -1) {
#ifndef GENERICIO_NO_MPI
    MPI_Comm_rank(Comm, &EffRank);
#else
    EffRank = 0;
#endif
  }

  openAndReadHeader(MismatchAllowed, EffRank, false);

  assert(FH.getHeaderCache().size() && "HeaderCache must not be empty");

  GlobalHeader<IsBigEndian> *GH = (GlobalHeader<IsBigEndian> *) &FH.getHeaderCache()[0];
  size_t RankIndex = getRankIndex<IsBigEndian>(EffRank, GH, RankMap, FH.getHeaderCache());

  assert(RankIndex < GH->NRanks && "Invalid rank specified");

  RankHeader<IsBigEndian> *RH = (RankHeader<IsBigEndian> *) &FH.getHeaderCache()[GH->RanksStart +
                                               RankIndex*GH->RanksSize];

  std::copy(RH->Coords, RH->Coords + 3, Coords);
}

void GenericIO::readData(int EffRank, bool PrintStats, bool CollStats) {
  int Rank;
#ifndef GENERICIO_NO_MPI
  MPI_Comm_rank(Comm, &Rank);
#else
  Rank = 0;
#endif

  uint64_t TotalReadSize = 0;
#ifndef GENERICIO_NO_MPI
  double StartTime = MPI_Wtime();
#else
  double StartTime = double(clock())/CLOCKS_PER_SEC;
#endif

  int NErrs[3] = { 0, 0, 0 };

  if (EffRank == -1 && Redistributing) {
    DisableCollErrChecking = true;

    size_t RowOffset = 0;
    for (int i = 0, ie = SourceRanks.size(); i != ie; ++i) {
      readData(SourceRanks[i], RowOffset, Rank, TotalReadSize, NErrs);
      RowOffset += readNumElems(SourceRanks[i]);
    }

    DisableCollErrChecking = false;
  } else {
    readData(EffRank, 0, Rank, TotalReadSize, NErrs);
  }

  int AllNErrs[3];
#ifndef GENERICIO_NO_MPI
  MPI_Allreduce(NErrs, AllNErrs, 3, MPI_INT, MPI_SUM, Comm);
#else
  AllNErrs[0] = NErrs[0]; AllNErrs[1] = NErrs[1]; AllNErrs[2] = NErrs[2];
#endif

  if (AllNErrs[0] > 0 || AllNErrs[1] > 0 || AllNErrs[2] > 0) {
    stringstream ss;
    ss << "Experienced " << AllNErrs[0] << " I/O error(s), " <<
          AllNErrs[1] << " CRC error(s) and " << AllNErrs[2] <<
          " decompression CRC error(s) reading: " << OpenFileName;
    throw runtime_error(ss.str());
  }

#ifndef GENERICIO_NO_MPI
  MPI_Barrier(Comm);
#endif

#ifndef GENERICIO_NO_MPI
  double EndTime = MPI_Wtime();
#else
  double EndTime = double(clock())/CLOCKS_PER_SEC;
#endif

  double TotalTime = EndTime - StartTime;
  double MaxTotalTime;
#ifndef GENERICIO_NO_MPI
  if (CollStats)
    MPI_Reduce(&TotalTime, &MaxTotalTime, 1, MPI_DOUBLE, MPI_MAX, 0, Comm);
  else
#endif
  MaxTotalTime = TotalTime;

  uint64_t AllTotalReadSize;
#ifndef GENERICIO_NO_MPI
  if (CollStats)
    MPI_Reduce(&TotalReadSize, &AllTotalReadSize, 1, MPI_UINT64_T, MPI_SUM, 0, Comm);
  else
#endif
  AllTotalReadSize = TotalReadSize;

  if (Rank == 0 && PrintStats) {
    double Rate = ((double) AllTotalReadSize) / MaxTotalTime / (1024.*1024.);
    cout << "Read " << Vars.size() << " variables from " << FileName <<
            " (" << AllTotalReadSize << " bytes) in " << MaxTotalTime << "s: " <<
            Rate << " MB/s [excluding header read]" << endl;
  }
}
#ifdef GENERICIO_HAVE_HDF
  void GenericIO::readDataHDF(int EffRank, bool PrintStats, bool CollStats) {
  int Rank, commRanks;

  std::vector<std::string> v;
#ifndef GENERICIO_NO_MPI
  MPI_Comm_rank(Comm, &Rank);
  MPI_Comm_size(MPI_COMM_WORLD, &commRanks);
#else
  Rank = 0;
#endif

  hid_t dset, attr_id, space, aspace;
  hid_t dset_CRC;
  hid_t fid, dtype,dxpl_id;
  herr_t ret;
  char *c_str = new char[20 + 1];
  char *c_str3 = new char[20 + 5];
  hsize_t dims[1]; // dataspace dim sizes
  hid_t dataset, file_dataspace, mem_dataspace;
  hid_t dataset_CRC, file_dataspace_CRC, mem_dataspace_CRC;
  hsize_t start[1];			/* for hyperslab setting */
  hsize_t hypercount[1], stride[1];	/* for hyperslab setting */
  hsize_t sizedims[1];
  uint64_t CRCv[1], CRC_loc;
  hsize_t dim_size[1];
  uint64_t read1_cs = 0;

  //   cout << Vars[7].IsFloat << " aaa " << Vars[7].IsPhysCoordX <<  endl;
  //  cout << Vars[8].Size << endl;
  // cout << "inside readData" << Vars.size() << endl;

  //  openAndReadHeader(false, EffRank, false);

//   assert(FH.getHeaderCache().size() && "HeaderCache must not be empty");

//   if (EffRank == -1)
//     EffRank = Rank;

//   GlobalHeader<IsBigEndian> *GH = (GlobalHeader<IsBigEndian> *) &FH.getHeaderCache()[0];
//   size_t RankIndex = getRankIndex<IsBigEndian>(EffRank, GH, RankMap, FH.getHeaderCache());

//   assert(RankIndex < GH->NRanks && "Invalid rank specified");

//   RankHeader<IsBigEndian> *RH = (RankHeader<IsBigEndian> *) &FH.getHeaderCache()[GH->RanksStart +
//                                                RankIndex*GH->RanksSize];

//   uint64_t TotalReadSize = 0;
// #ifndef GENERICIO_NO_MPI
//   double StartTime = MPI_Wtime();
// #else
//   double StartTime = double(clock())/CLOCKS_PER_SEC;
// #endif

//   int NErrs[3] = { 0, 0, 0 };

   v.push_back("/Variables/id");
   v.push_back("/Variables/mask");
   v.push_back("/Variables/phi");
   v.push_back("/Variables/vx");
   v.push_back("/Variables/vy");
   v.push_back("/Variables/vz");
   v.push_back("/Variables/x");
   v.push_back("/Variables/y");
   v.push_back("/Variables/z");

   GenericFileIO_HDF *gfio_hdf = dynamic_cast<GenericFileIO_HDF *> (FH.get());
   fid = gfio_hdf->get_fileid();
   dims[0] = gfio_hdf->get_NumElem();
   //cout << dims[0] << endl;


    sizedims[0] = 1;
    mem_dataspace_CRC = H5Screate_simple (1, sizedims, NULL);

   for (size_t i = 0; i < Vars.size(); ++i) {
//     uint64_t Offset = RH->Start;
     bool VarFound = false;

     Vars[i].Name = v[i];

     std::strcpy(c_str, v[i].c_str());

     // cout << c_str << endl;
     // cout << fid << endl;
     dset = H5Dopen(fid, c_str, H5P_DEFAULT);


     file_dataspace = H5Dget_space (dset);

     H5Sget_simple_extent_dims(file_dataspace, dim_size, NULL);


//      //     FH.get()->read(Data, ReadSize, Offset, Vars[i].Name);

     start[0] =  Rank*dim_size[0]/commRanks;
     hypercount[0] = dims[0];
     stride[0] = 1;

     // cout << start[0] << endl;
     // cout <<  hypercount[0] << endl;

     ret=H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, start, stride,
 	    hypercount, NULL);

     sizedims[0] = dims[0];

     mem_dataspace = H5Screate_simple (1, sizedims, NULL);

     // Vars[0].Data[0] = 22;

     void *Data = Vars[i].Data;

     if (!Data) cout << "NULLISH" << endl;

     dxpl_id = H5Pcreate (H5P_DATASET_XFER);
     //H5Pset_dxpl_checksum_ptr(dxpl_id, &read1_cs);

     if(Rank==0) printf("Reading Dataset  %s \n",c_str);
     if( Vars[i].Name.compare("/Variables/id") == 0) {
       dtype = H5T_NATIVE_LONG;
     } else if( Vars[i].Name.compare("/Variables/mask") == 0) {
       dtype = H5T_NATIVE_B16;
     } else  {
       dtype = H5T_NATIVE_FLOAT;
     }
     ret = H5Dread(dset, dtype, mem_dataspace, file_dataspace, dxpl_id, Data);
     
     cout << "H5Dread" << ret << endl;

     H5Pclose(dxpl_id);

     ret = H5Dclose (dset);
     cout << fid << endl;

     ret = H5Sclose (mem_dataspace);
     cout << "100strcpy" << ret << endl;
     ret = H5Sclose (file_dataspace);
     cout << "99strcpy" << ret << endl;

     cout << "3strcpy" << ret << endl;
      std::strcpy(c_str3, c_str);
     cout << "3asdfstrcpy" << ret << endl;
      strcat(c_str3, "_CRC");

     cout << "2strcpy" << ret << endl;
      if(Rank==0) printf("Reading in CRC for Dataset %s \n",c_str3);

      attr_id = H5Dopen(fid, c_str3, H5P_DEFAULT);
      file_dataspace_CRC = H5Dget_space(attr_id);

      start[0] = Rank;
      ret=H5Sselect_elements(file_dataspace_CRC, H5S_SELECT_SET, 1, start);

      ret = H5Dread(attr_id, H5T_NATIVE_ULONG, mem_dataspace_CRC, file_dataspace_CRC, H5P_DEFAULT, CRCv);

//     if(Vars[i].Name.compare("/Variables/phi") == 0) {
//       for(size_t j = 0; j < dims[0]; ++j){
// 	cout << j << " " << ((float*)Data)[j] << endl;
//       }
//     }
      CRC_loc = crc64_omp(Vars[i].Data, dims[0]);

      ret = H5Dclose(attr_id);
      ret = H5Sclose(file_dataspace_CRC);
      
      if(Rank==0) printf("Checking in CRC for Dataset %s ",c_str3);
      if (CRCv[0] != CRC_loc) {
	
        cout << "CRC error " << CRCv[0] << " " << CRC_loc << endl;
      } else {
	
      if(Rank==0) printf("PASSED \n",c_str3);
      }

   }

   
   //   ret = H5Pclose(fapl_id);

   
   //ret=H5Fclose(fid);
   MPI_Barrier(MPI_COMM_WORLD);
   // cout << "after close" << endl;

   delete[] c_str;
   delete[] c_str3;

}
#endif

void GenericIO::readData(int EffRank, size_t RowOffset, int Rank,
                         uint64_t &TotalReadSize, int NErrs[3]) {
  if (FH.isBigEndian())
    readData<true>(EffRank, RowOffset, Rank, TotalReadSize, NErrs);
  else
    readData<false>(EffRank, RowOffset, Rank, TotalReadSize, NErrs);
}

// Note: Errors from this function should be recoverable. This means that if
// one rank throws an exception, then all ranks should.
template <bool IsBigEndian>
void GenericIO::readData(int EffRank, size_t RowOffset, int Rank,
                         uint64_t &TotalReadSize, int NErrs[3]) {
  openAndReadHeader(Redistributing ? MismatchRedistribute : MismatchAllowed,
                    EffRank, false);

  assert(FH.getHeaderCache().size() && "HeaderCache must not be empty");

  if (EffRank == -1)
    EffRank = Rank;

  GlobalHeader<IsBigEndian> *GH = (GlobalHeader<IsBigEndian> *) &FH.getHeaderCache()[0];
  size_t RankIndex = getRankIndex<IsBigEndian>(EffRank, GH, RankMap, FH.getHeaderCache());

  assert(RankIndex < GH->NRanks && "Invalid rank specified");

  RankHeader<IsBigEndian> *RH = (RankHeader<IsBigEndian> *) &FH.getHeaderCache()[GH->RanksStart +
                                               RankIndex*GH->RanksSize];

  for (size_t i = 0; i < Vars.size(); ++i) {
    uint64_t Offset = RH->Start;
    bool VarFound = false;
    for (uint64_t j = 0; j < GH->NVars; ++j) {
      VariableHeader<IsBigEndian> *VH = (VariableHeader<IsBigEndian> *) &FH.getHeaderCache()[GH->VarsStart +
                                                           j*GH->VarsSize];

      string VName(VH->Name, VH->Name + NameSize);
      size_t VNameNull = VName.find('\0');
      if (VNameNull < NameSize)
        VName.resize(VNameNull);

      uint64_t ReadSize = RH->NElems*VH->Size + CRCSize;
      if (VName != Vars[i].Name) {
        Offset += ReadSize;
        continue;
      }

      size_t ElementSize = VH->Size;
      if (offsetof_safe(VH, ElementSize) < GH->VarsSize)
        ElementSize = VH->ElementSize;

      VarFound = true;
      bool IsFloat = (bool) (VH->Flags & FloatValue),
           IsSigned = (bool) (VH->Flags & SignedValue);
      if (VH->Size != Vars[i].Size) {
        stringstream ss;
        ss << "Size mismatch for variable " << Vars[i].Name <<
              " in: " << OpenFileName << ": current: " << Vars[i].Size <<
              ", file: " << VH->Size;
        throw runtime_error(ss.str());
      } else if (ElementSize != Vars[i].ElementSize) {
        stringstream ss;
        ss << "Element size mismatch for variable " << Vars[i].Name <<
              " in: " << OpenFileName << ": current: " << Vars[i].ElementSize <<
              ", file: " << ElementSize;
        throw runtime_error(ss.str());
      } else if (IsFloat != Vars[i].IsFloat) {
        string Float("float"), Int("integer");
        stringstream ss;
        ss << "Type mismatch for variable " << Vars[i].Name <<
              " in: " << OpenFileName << ": current: " <<
              (Vars[i].IsFloat ? Float : Int) <<
              ", file: " << (IsFloat ? Float : Int);
        throw runtime_error(ss.str());
      } else if (IsSigned != Vars[i].IsSigned) {
        string Signed("signed"), Uns("unsigned");
        stringstream ss;
        ss << "Type mismatch for variable " << Vars[i].Name <<
              " in: " << OpenFileName << ": current: " <<
              (Vars[i].IsSigned ? Signed : Uns) <<
              ", file: " << (IsSigned ? Signed : Uns);
        throw runtime_error(ss.str());
      }

      size_t VarOffset = RowOffset*Vars[i].Size;
      void *VarData = ((char *) Vars[i].Data) + VarOffset;

      vector<unsigned char> LData;
      void *Data = VarData;
      bool HasExtraSpace = Vars[i].HasExtraSpace;
      if (offsetof_safe(GH, BlocksStart) < GH->GlobalHeaderSize &&
          GH->BlocksSize > 0) {
        BlockHeader<IsBigEndian> *BH = (BlockHeader<IsBigEndian> *)
          &FH.getHeaderCache()[GH->BlocksStart +
                               (RankIndex*GH->NVars + j)*GH->BlocksSize];
        ReadSize = BH->Size + CRCSize;
        Offset = BH->Start;

        if (strncmp(BH->Filters[0], CompressName, FilterNameSize) == 0) {
          LData.resize(ReadSize);
          Data = &LData[0];
          HasExtraSpace = true;
        } else if (BH->Filters[0][0] != '\0') {
          stringstream ss;
          ss << "Unknown filter \"" << BH->Filters[0] << "\" on variable " << Vars[i].Name;
          throw runtime_error(ss.str());
        }
      }

      assert(HasExtraSpace && "Extra space required for reading");

      char CRCSave[CRCSize];
      char *CRCLoc = ((char *) Data) + ReadSize - CRCSize;
      if (HasExtraSpace)
        std::copy(CRCLoc, CRCLoc + CRCSize, CRCSave);

      int Retry = 0;
      {
        int RetryCount = 300;
        const char *EnvStr = getenv("GENERICIO_RETRY_COUNT");
        if (EnvStr)
          RetryCount = atoi(EnvStr);

        int RetrySleep = 100; // ms
        EnvStr = getenv("GENERICIO_RETRY_SLEEP");
        if (EnvStr)
          RetrySleep = atoi(EnvStr);

        for (; Retry < RetryCount; ++Retry) {
          try {
            FH.get()->read(Data, ReadSize, Offset, Vars[i].Name);
            break;
          } catch (...) { }

          usleep(1000*RetrySleep);
        }

        if (Retry == RetryCount) {
          ++NErrs[0];
          break;
        } else if (Retry > 0) {
          EnvStr = getenv("GENERICIO_VERBOSE");
          if (EnvStr) {
            int Mod = atoi(EnvStr);
            if (Mod > 0) {
              int Rank;
#ifndef GENERICIO_NO_MPI
              MPI_Comm_rank(MPI_COMM_WORLD, &Rank);
#else
              Rank = 0;
#endif

              std::cerr << "Rank " << Rank << ": " << Retry <<
                           " I/O retries were necessary for reading " <<
                           Vars[i].Name << " from: " << OpenFileName << "\n";

              std::cerr.flush();
            }
          }
        }
      }

      TotalReadSize += ReadSize;

      uint64_t CRC = crc64_omp(Data, ReadSize);
      if (CRC != (uint64_t) -1) {
        ++NErrs[1];

        int Rank;
#ifndef GENERICIO_NO_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &Rank);
#else
        Rank = 0;
#endif

        // All ranks will do this and have a good time!
        string dn = "gio_crc_errors";
        mkdir(dn.c_str(), 0777);

        srand(time(0));
        int DumpNum = rand();
        stringstream ssd;
        ssd << dn << "/gio_crc_error_dump." << Rank << "." << DumpNum << ".bin";

        stringstream ss;
        ss << dn << "/gio_crc_error_log." << Rank << ".txt";

        ofstream ofs(ss.str().c_str(), ofstream::out | ofstream::app);
        ofs << "On-Disk CRC Error Report:\n";
        ofs << "Variable: " << Vars[i].Name << "\n";
        ofs << "File: " << OpenFileName << "\n";
        ofs << "I/O Retries: " << Retry << "\n";
        ofs << "Size: " << ReadSize << " bytes\n";
        ofs << "Offset: " << Offset << " bytes\n";
        ofs << "CRC: " << CRC << " (expected is -1)\n";
        ofs << "Dump file: " << ssd.str() << "\n";
        ofs << "\n";
        ofs.close();

        ofstream dofs(ssd.str().c_str(), ofstream::out);
        dofs.write((const char *) Data, ReadSize);
        dofs.close();

        uint64_t RawCRC = crc64_omp(Data, ReadSize - CRCSize);
        unsigned char *UData = (unsigned char *) Data;
        crc64_invert(RawCRC, &UData[ReadSize - CRCSize]);
        uint64_t NewCRC = crc64_omp(Data, ReadSize);
        std::cerr << "Recalulated CRC: " << NewCRC << ((NewCRC == -1) ? "ok" : "bad") << "\n";
        break;
      }

      if (HasExtraSpace)
        std::copy(CRCSave, CRCSave + CRCSize, CRCLoc);

      if (LData.size()) {
        CompressHeader<IsBigEndian> *CH = (CompressHeader<IsBigEndian>*) &LData[0];

#ifdef _OPENMP
#pragma omp master
  {
#endif

       if (!blosc_initialized) {
         blosc_init();
         blosc_initialized = true;
       }

#ifdef _OPENMP
       blosc_set_nthreads(omp_get_max_threads());
  }
#endif

        blosc_decompress(&LData[0] + sizeof(CompressHeader<IsBigEndian>),
                         VarData, Vars[i].Size*RH->NElems);

        if (CH->OrigCRC != crc64_omp(VarData, Vars[i].Size*RH->NElems)) {
          ++NErrs[2];
          break;
        }
      }

      // Byte swap the data if necessary.
      if (IsBigEndian != isBigEndian())
        for (size_t j = 0;
             j < RH->NElems*(Vars[i].Size/Vars[i].ElementSize); ++j) {
          char *Offset = ((char *) VarData) + j*Vars[i].ElementSize;
          bswap(Offset, Vars[i].ElementSize);
        }

      break;
    }

    if (!VarFound)
      throw runtime_error("Variable " + Vars[i].Name +
                          " not found in: " + OpenFileName);

    // This is for debugging.
    if (NErrs[0] || NErrs[1] || NErrs[2]) {
      const char *EnvStr = getenv("GENERICIO_VERBOSE");
      if (EnvStr) {
        int Mod = atoi(EnvStr);
        if (Mod > 0) {
          int Rank;
#ifndef GENERICIO_NO_MPI
          MPI_Comm_rank(MPI_COMM_WORLD, &Rank);
#else
          Rank = 0;
#endif

          std::cerr << "Rank " << Rank << ": " << NErrs[0] << " I/O error(s), " <<
          NErrs[1] << " CRC error(s) and " << NErrs[2] <<
          " decompression CRC error(s) reading: " << Vars[i].Name <<
          " from: " << OpenFileName << "\n";

          std::cerr.flush();
        }
      }
    }

    if (NErrs[0] || NErrs[1] || NErrs[2])
      break;
  }
}

void GenericIO::getVariableInfo(vector<VariableInfo> &VI) {
  if (FH.isBigEndian())
    getVariableInfo<true>(VI);
  else
    getVariableInfo<false>(VI);
}

template <bool IsBigEndian>
void GenericIO::getVariableInfo(vector<VariableInfo> &VI) {
  assert(FH.getHeaderCache().size() && "HeaderCache must not be empty");

  GlobalHeader<IsBigEndian> *GH = (GlobalHeader<IsBigEndian> *) &FH.getHeaderCache()[0];
  for (uint64_t j = 0; j < GH->NVars; ++j) {
    VariableHeader<IsBigEndian> *VH = (VariableHeader<IsBigEndian> *) &FH.getHeaderCache()[GH->VarsStart +
                                                         j*GH->VarsSize];

    string VName(VH->Name, VH->Name + NameSize);
    size_t VNameNull = VName.find('\0');
    if (VNameNull < NameSize)
      VName.resize(VNameNull);

    size_t ElementSize = VH->Size;
    if (offsetof_safe(VH, ElementSize) < GH->VarsSize)
      ElementSize = VH->ElementSize;

    bool IsFloat = (bool) (VH->Flags & FloatValue),
         IsSigned = (bool) (VH->Flags & SignedValue),
         IsPhysCoordX = (bool) (VH->Flags & ValueIsPhysCoordX),
         IsPhysCoordY = (bool) (VH->Flags & ValueIsPhysCoordY),
         IsPhysCoordZ = (bool) (VH->Flags & ValueIsPhysCoordZ),
         MaybePhysGhost = (bool) (VH->Flags & ValueMaybePhysGhost);
    VI.push_back(VariableInfo(VName, (size_t) VH->Size, IsFloat, IsSigned,
                              IsPhysCoordX, IsPhysCoordY, IsPhysCoordZ,
                              MaybePhysGhost, ElementSize));
  }
}

void GenericIO::setNaturalDefaultPartition() {
#ifdef __bgq__
  DefaultPartition = MPIX_IO_link_id();
#else
#ifndef GENERICIO_NO_MPI
  bool UseName = true;
  const char *EnvStr = getenv("GENERICIO_PARTITIONS_USE_NAME");
  if (EnvStr) {
    int Mod = atoi(EnvStr);
    UseName = (Mod != 0);
  }

  if (UseName) {
    // This is a heuristic to generate ~256 partitions based on the
    // names of the nodes.
    char Name[MPI_MAX_PROCESSOR_NAME];
    int Len = 0;

    MPI_Get_processor_name(Name, &Len);
    unsigned char color = 0;
    for (int i = 0; i < Len; ++i)
      color += (unsigned char) Name[i];

    DefaultPartition = color;
  }

  // This is for debugging.
  EnvStr = getenv("GENERICIO_RANK_PARTITIONS");
  if (EnvStr) {
    int Mod = atoi(EnvStr);
    if (Mod > 0) {
      int Rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &Rank);
      DefaultPartition += Rank % Mod;
    }
  }
#endif
#endif
}

} /* END namespace cosmotk */
