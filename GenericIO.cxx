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
 * THE SOFTWARE IS SUPPLIED %Gâ€œ%@AS IS-fµ WITHOUT WARRANTY OF ANY KIND.  NEITHER THE-A
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

#define HDF5_COMPRESSION
#define HDF5_COMPRESSION_INDIVIDUAL_IO
//#define HDF5_HAVE_MULTI_DATASETS
#ifdef HDF5_HAVE_MULTI_DATASETS
H5D_rw_multi_t multi_info[9];
#endif

// COMPOUND TYPE METHOD
#define HDF5_DERV
#ifdef HDF5_DERV
typedef struct {
  int64_t id;
  uint16_t mask;
  float   x;
  float   y;
  float   z;
  float   vx;
  float   vy;
  float   vz;;
  float   phi;
} hacc_t;
hacc_t *Hdata;
hid_t Hmemtype;
hid_t Hfiletype;

uint64_t CRC_values[9];

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
  hid_t fcpl_id;
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
  //MPI_Comm_rank(MPI_COMM_WORLD, &commRank);
  FileName = FN;

  MPI_Info info;
  MPI_Info_create(&info);
  MPI_Info_set(info, "cb_nodes","4");

  // setup file access template with parallel IO access.
  fapl_id = H5Pcreate (H5P_FILE_ACCESS);
  //ret = H5Pset_fapl_mpiposix(fapl_id, Comm, 0); 
  ret = H5Pset_fapl_mpio(fapl_id, Comm, info);
  H5Pset_coll_metadata_write(fapl_id, 1);
  H5Pset_all_coll_metadata_ops(fapl_id, 1);
  H5Pset_libver_bounds(fapl_id, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST);
  H5Pset_fclose_degree(fapl_id,H5F_CLOSE_WEAK);
  fcpl_id = H5Pcreate(H5P_FILE_CREATE);  
//  H5Pset_file_space_strategy(fcpl_id,H5F_FSPACE_STRATEGY_PAGE,0,(hsize_t)1);
//  H5Pset_file_space_page_size(fcpl_id, (hsize_t)(2*1048576));
#if 0 
  H5AC_cache_config_t config;
  H5Pget_mdc_config(fapl_id, &config); 
  config.set_initial_size = 1;
  config.initial_size = 8388608;
  config.max_size = 12582912;
  config.min_size = 6291456;
  config.dirty_bytes_threshold = 12582912; 
  H5Pset_mdc_config(fapl_id, &config);
#endif   
#if 0
  H5Pset_coll_metadata_write(fapl_id, 1);
  H5Pset_all_coll_metadata_ops(fapl_id, 1 );
#endif
  FileName = FN;


  if ( ForReading) {
    if( (fid = H5Fopen(const_cast<char *>(FileName.c_str()),H5F_ACC_RDONLY,fapl_id)) < 0)
      throw runtime_error( ("Unable to open the file: ") + FileName);
  } else {
    if( (fid = H5Fcreate(const_cast<char *>(FileName.c_str()),H5F_ACC_TRUNC,fcpl_id,fapl_id)) < 0)
      throw runtime_error( ("Unable to create the file: ") + FileName);
  }
  /* Get read context */
  MPI_Info_free(&info);  
  FH=fid;

  //  if( (fid = H5Fopen(const_cast<char *>(FileName.c_str()),H5F_ACC_RDWR,fapl_id)) < 0 ) {    
  // Release file-access template
  ret = H5Pclose(fapl_id);
  ret = H5Pclose(fcpl_id);

  if ( ForReading ) {
    // get size
#ifdef HDF5_DERV
    dset  = H5Dopen (fid, "/Variables/DATA", H5P_DEFAULT);
#else
    dset  = H5Dopen (fid, "/Variables/id", H5P_DEFAULT);
#endif
    space = H5Dget_space (dset);
    ndims = H5Sget_simple_extent_dims (space, dims, NULL);

    ret = H5Dclose (dset);
    ret = H5Sclose (space);

    int commRanks;
    int Rank;
    MPI_Comm_size(MPI_COMM_WORLD, &commRanks);
    MPI_Comm_rank(MPI_COMM_WORLD, &Rank);    // MSB, why does comm not work?

    NumElem = dims[0]/commRanks;

    if(Rank+1 <= dims[0]%commRanks)
      NumElem +=1;

  }

  //  cout << "done open" << endl;
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

void GenericFileIO_HDF::write_hdf_internal(const void *buf, size_t count, uint64_t offset,
				  const std::string &D, hid_t dtype, hsize_t numel, hsize_t chunk_size,const void *CRC, hid_t gid, uint64_t Totnumel, size_t i) {

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
  double t1, timer;
  MPI_Comm_rank(MPI_COMM_WORLD, &commRank);
  MPI_Comm_size(MPI_COMM_WORLD, &commRanks); 

  adims[0] = (hsize_t)commRanks;
  dims[0] = (hsize_t)Totnumel;
 

  // cout << "inside hdf_write" << endl;

  // --------------------------
  // Define the dimensions of the overall datasets
  // and create the dataset
  // -------------------------
  // setup dimensionality object

  std::strcpy(c_str, D.c_str());


  if( (sid = H5Screate_simple (1, dims, NULL)) < 0)
      throw runtime_error( "Unable to create HDF5 space: " );


#ifndef HDF5_HAVE_MULTI_DATASETS

#ifdef HDF5_COMPRESSION
  hid_t dset_creation_plist;
  hsize_t chunk_dims1[1];
  chunk_dims1[0] = chunk_size; 
  dset_creation_plist = H5Pcreate(H5P_DATASET_CREATE);

  H5Pset_chunk(dset_creation_plist,1,chunk_dims1);

  H5Pset_deflate(dset_creation_plist,6);
 
  if( (dataset = H5Dcreate2(gid, c_str, dtype, sid, H5P_DEFAULT, dset_creation_plist, H5P_DEFAULT)) < 0)
    throw runtime_error( "Unable to create HDF5 dataset " );
  H5Pclose(dset_creation_plist);

#else  
  if( (dataset = H5Dcreate2(gid, c_str, dtype, sid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT)) < 0)
    throw runtime_error( "Unable to create HDF5 dataset " );
#endif

  delete[] c_str;

  std::strcpy(c_str3, D.c_str());
  strcat(c_str3, "_CRC");
 
  /* set up dimensions of the slab this process accesses */

  start[0] = offset; // commRank*Totnumel/commRanks;

  hypercount[0] = numel;

  // cout << commRank << " " << start[0] << endl;
  //cout << commRank << "count " << hypercount[0] << endl;

  stride[0] = 1;

  file_dataspace = H5Dget_space (dataset);
   
  ret=H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, start, stride,
			  hypercount, NULL);

  mem_dataspace = H5Screate_simple (1, hypercount, NULL);

  hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);

#ifdef HDF5_COMPRESSION
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
#ifdef HDF5_COMPRESSION_INDIVIDUAL_IO
  H5Pset_dxpl_mpio_collective_opt(plist_id,H5FD_MPIO_INDIVIDUAL_IO);
#endif
#endif
  // write data
  t1 = MPI_Wtime();

  ret = H5Dwrite(dataset, dtype, mem_dataspace, file_dataspace,
  		 plist_id, (void *)buf);
  
#else
  multi_info[i].mem_type_id = dtype;
  if( (multi_info[i].dset_id = H5Dcreate2(gid, c_str,  multi_info[i].mem_type_id, sid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT)) < 0)
    throw runtime_error( "Unable to create HDF5 dataset " );

  delete[] c_str;

  std::strcpy(c_str3, D.c_str());
  strcat(c_str3, "_CRC");
 
  /* set up dimensions of the slab this process accesses */

  start[0] = offset; // commRank*Totnumel/commRanks;

  hypercount[0] = numel;

  // cout << commRank << " " << start[0] << endl;
  //cout << commRank << "count " << hypercount[0] << endl;

  stride[0] = 1;

  multi_info[i].dset_space_id = H5Dget_space (multi_info[i].dset_id);
   
  ret=H5Sselect_hyperslab( multi_info[i].dset_space_id, H5S_SELECT_SET, start, stride,
			  hypercount, NULL);

  multi_info[i].mem_space_id = H5Screate_simple (1, hypercount, NULL);

  multi_info[i].u.wbuf = buf;
#endif

  timer = MPI_Wtime()-t1;
#if 0
  double *rtimers=NULL;
  if(commRank == 0)  {
    H5D_mpio_actual_io_mode_t actual_io_mode;
    H5Pget_mpio_actual_io_mode( plist_id, &actual_io_mode);
    cout << " collective mode " << actual_io_mode << endl; 
    rtimers = (double *) malloc(commRanks*sizeof(double));
  }

  MPI_Gather(&timer, 1, MPI_DOUBLE, rtimers, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if(commRank == 0)  {
    double mean=0;
    for(int n = 0; n < commRanks; n++) {
      mean += rtimers[n];
    }
    printf("H5DRead = %.2f \n", mean/commRanks);
    free(rtimers);
  }
#endif
  // release dataspaceID
#ifndef HDF5_HAVE_MULTI_DATASETS
  H5Pclose (plist_id);
  H5Sclose (file_dataspace);
  H5Sclose (mem_dataspace);
  ret=H5Dclose(dataset);
#endif
  H5Sclose (sid);

  //
  // Create the CRC dataset
  //

#if 1
  file_dataspace = H5Screate(H5S_SCALAR);
  dataset = H5Dcreate2(gid, c_str3, H5T_NATIVE_ULONG, file_dataspace,
			H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  delete[] c_str3;
  if( commRank == 0 ) {
    mem_dataspace = H5Screate(H5S_SCALAR);
    ret = H5Dwrite(dataset, H5T_NATIVE_ULONG, mem_dataspace, file_dataspace,
		   H5P_DEFAULT, (void *)CRC);
  } else {
    mem_dataspace = H5Screate(H5S_NULL);
    ret = H5Dwrite(dataset, H5T_NATIVE_ULONG, mem_dataspace, mem_dataspace,
		   H5P_DEFAULT, NULL);
  }
  H5Sclose (mem_dataspace);
  H5Sclose (file_dataspace);
  H5Dclose (dataset);
#endif 
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

void GenericIO::write_hdf() {

  hid_t fid, space, dset, attr, filetype, atype, gid, aid, tid1, rid1, rid2, dxpl_id;
  hid_t fapl_id;  // File access templates
  hid_t sid, dataset;
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
  double timestep[1];
  char notes[] = {"Important notes go here"};

  uint64_t FileSize = 0;
  hsize_t chunk_size = 0;

  int NRanks, Rank;
  MPI_Comm_rank(Comm, &Rank);
  MPI_Comm_size(Comm, &NRanks);

#ifdef __bgq__
  MPI_Barrier(Comm);
#endif


  double StartTime = MPI_Wtime();

  GenericFileIO_HDF *gfio_hdf;
  // The communicator may be COMM_SELF for independent IO?
  FH.get() = new GenericFileIO_HDF(MPI_COMM_WORLD);
  FH.get()->open(FileName);

  if (FileIOType == FileIOHDF) {// Can remove this line later
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

#ifdef HDF5_COMPRESSION
    uint64_t max_nelms = 0;
    MPI_Allreduce(&NElems, &max_nelms, 1, MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);
    //assert(((uint64_t)((size_t)(max_nelms)) == max_nelms);
    chunk_size = (hsize_t)max_nelms;

#endif
  } // Can remove this line later


#ifdef HDF5_DERV
  Hdata = (hacc_t *) malloc (NElems * sizeof (hacc_t));

  Hmemtype = H5Tcreate (H5T_COMPOUND, sizeof (hacc_t));
  H5Tinsert (Hmemtype, "id",
		      HOFFSET (hacc_t, id), H5T_NATIVE_LONG);
  H5Tinsert (Hmemtype, "mask", 
		      HOFFSET (hacc_t, mask), H5T_NATIVE_UINT16);
  H5Tinsert (Hmemtype, "x",
		      HOFFSET (hacc_t, x), H5T_NATIVE_FLOAT);
  H5Tinsert (Hmemtype, "y",
		      HOFFSET (hacc_t, y), H5T_NATIVE_FLOAT);
  H5Tinsert (Hmemtype, "z",
		      HOFFSET (hacc_t, z), H5T_NATIVE_FLOAT);
  H5Tinsert (Hmemtype, "vx",
		      HOFFSET (hacc_t, vx), H5T_NATIVE_FLOAT);
  H5Tinsert (Hmemtype, "vy",
		      HOFFSET (hacc_t, vy), H5T_NATIVE_FLOAT);
  H5Tinsert (Hmemtype, "vz",
		      HOFFSET (hacc_t, vz), H5T_NATIVE_FLOAT);
  H5Tinsert (Hmemtype, "phi",
		      HOFFSET (hacc_t, phi), H5T_NATIVE_FLOAT);

#endif

  uint64_t Offsets_glb;
	
  double t1, timer=0.;
  for (size_t i = 0; i < Vars.size(); ++i) {

    uint64_t WriteSize = NElems*Vars[i].Size;
    void *Data = Vars[i].Data;
    uint64_t CRC = 0;

    if (FileIOType == FileIOHDF) {
	  hid_t dtype;
	  // CRC calculation 
	  crc send;
	  //	cout << Vars[i].Size << Vars[i].Name << endl;
	  send.CRC64 = crc64_omp(Data, NElems*Vars[i].Size);
	  send.CRC64_size = NElems*Vars[i].Size;
	  struct crc_s *rbufv;
	  uint64_t *sbufv;

	  int          blocklengths[2] = {1,1};
  	  MPI_Datatype types[2] = {MPI_UINT64_T, MPI_LONG_LONG_INT };
	  MPI_Datatype mpi_crc_type;
	  MPI_Aint     offsets[2];

	  offsets[0] = offsetof(crc, CRC64);
	  offsets[1] = offsetof(crc, CRC64_size);

	  MPI_Type_create_struct(2, blocklengths, offsets, types, &mpi_crc_type);
	  MPI_Type_commit(&mpi_crc_type);

	  rbufv = NULL;
	  if(Rank == 0) {
	    rbufv = new crc_s [NRanks*sizeof(struct crc_s)];
	  }
	  sbufv = new uint64_t [NRanks*sizeof(uint64_t)];

	  MPI_Gather( &send, 1, mpi_crc_type, rbufv, 1, mpi_crc_type, 0, MPI_COMM_WORLD);
	  if(Rank == 0) {
	    uint64_t CRC_sum = 0;
	    sbufv[0] = 0;
	    int k;
	    for (k =0; k<NRanks; k++) {
	      CRC_sum = crc64_combine(CRC_sum, rbufv[k].CRC64, rbufv[k].CRC64_size);
	      // find offsets
	      if(k > 0)
	        sbufv[k] = sbufv[k-1] + rbufv[k-1].CRC64_size/Vars[i].Size;
	      // cout << k << " : " << sbufv[k] << endl;
	    }
	    delete rbufv;
	    CRC = CRC_sum;
	    //  cout << "CRC_sum" << CRC << endl;
	  } else
	    CRC = 0;

	  uint64_t Offsets;
      /// What does this line do??
	  MPI_Scatter( sbufv, 1, MPI_UINT64_T, &Offsets, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD); 
	  delete sbufv;
	  MPI_Type_free(&mpi_crc_type);

#ifdef HDF5_DERV
	  hsize_t ii;
	  if( Vars[i].Name.compare("id") == 0) {
	    for (ii=0; ii < NElems; ii++)
	      Hdata[ii].id = *((int64_t *)Data + ii);
	    CRC_values[0] = CRC;
	  } else if( Vars[i].Name.compare("mask") == 0) {
	    for (ii=0; ii < NElems; ii++)
	      Hdata[ii].mask = *((uint16_t *)Data + ii);
	    CRC_values[1] = CRC;
	  } else if( Vars[i].Name.compare("x") == 0) {
	    for (ii=0; ii < NElems; ii++)
	      Hdata[ii].x = *((float *)Data + ii);
	    CRC_values[2] = CRC;
	  } else if( Vars[i].Name.compare("y") == 0) {
	    for (ii=0; ii < NElems; ii++)
	      Hdata[ii].y = *((float *)Data + ii);
	    CRC_values[3] = CRC;
	  } else if( Vars[i].Name.compare("z") == 0) {
	    for (ii=0; ii < NElems; ii++)
	      Hdata[ii].z = *((float *)Data + ii);
	    CRC_values[4] = CRC;
	  } else if( Vars[i].Name.compare("vx") == 0) {
	    for (ii=0; ii < NElems; ii++)
	      Hdata[ii].vx = *((float *)Data + ii);
	    CRC_values[5] = CRC;
	  } else if( Vars[i].Name.compare("vy") == 0) {
	    for (ii=0; ii < NElems; ii++)
	      Hdata[ii].vy = *((float *)Data + ii);
	    CRC_values[6] = CRC;
	  } else if( Vars[i].Name.compare("vz") == 0) {
	    for (ii=0; ii < NElems; ii++)
	      Hdata[ii].vz = *((float *)Data + ii);
	    CRC_values[7] = CRC;
	  } else if( Vars[i].Name.compare("phi") == 0) {
	    for (ii=0; ii < NElems; ii++) 
	      Hdata[ii].phi = *((float *)Data + ii);
	    CRC_values[8] = CRC;
	  }
	
	  Offsets_glb = Offsets;

#else
	  if( Vars[i].Name.compare("id") == 0) {
	    dtype = H5T_NATIVE_LONG;
	  } else if( Vars[i].Name.compare("mask") == 0) {
	    dtype = H5T_NATIVE_UINT16;
	  }else {
	    dtype = H5T_NATIVE_FLOAT;
	  }
	
	  gfio_hdf->write_hdf_internal(Data, WriteSize, Offsets , Vars[i].Name, dtype, NElems, chunk_size,&CRC, gid, TotElem, i);
#endif
    }
  }

#ifdef HDF5_DERV
  if (FileIOType == FileIOHDF) {
   /*
     * Create dataspace.  Setting maximum size to NULL sets the maximum
     * size to be the current size.
     */
    hid_t filespace, memspace, dset, plist_id;
    hsize_t     dims[1];
    dims[0] = (hsize_t)TotElem;
    filespace = H5Screate_simple (1, dims, NULL);

    /*
     * Create the dataset and write the compound data to it.
     */
#ifdef HDF5_COMPRESSION
  hid_t dset_creation_plist;
  hsize_t chunk_dims1[1];
  chunk_dims1[0] = chunk_size; 
  dset_creation_plist = H5Pcreate(H5P_DATASET_CREATE);

  H5Pset_chunk(dset_creation_plist,1,chunk_dims1);

  H5Pset_deflate(dset_creation_plist,6);
 
  //if( (dataset = H5Dcreate2(gid, c_str, dtype, sid, H5P_DEFAULT, dset_creation_plist, H5P_DEFAULT)) < 0)
  if((dset = H5Dcreate (gid, "DATA", Hmemtype, filespace, H5P_DEFAULT, dset_creation_plist, H5P_DEFAULT))<0)
     throw runtime_error( "Unable to create HDF5 dataset " );
  H5Pclose(dset_creation_plist);


#else
    dset = H5Dcreate (gid, "DATA", Hmemtype, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
#endif

    hsize_t count[1];	          /* hyperslab selection parameters */
    hsize_t offset[1];
    herr_t status;
    count[0] = NElems;
    memspace = H5Screate_simple(1, count, NULL);

    offset[0] = Offsets_glb;
//cerr<<"mpi_rank is "<<Rank <<" count: "<<count[0] <<"offset "<<offset[0]<<endl;

    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);

    plist_id = H5Pcreate(H5P_DATASET_XFER);
#ifdef HDF5_COMPRESSION
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
#ifdef HDF5_COMPRESSION_INDIVIDUAL_IO
  H5Pset_dxpl_mpio_collective_opt(plist_id,H5FD_MPIO_INDIVIDUAL_IO);
#endif
#endif

    t1 = MPI_Wtime();
    status = H5Dwrite (dset, Hmemtype, memspace, filespace, plist_id, Hdata);
    timer = MPI_Wtime()-t1;

    H5Pclose(plist_id);
    status = H5Dclose (dset);
    status = H5Sclose (filespace);
    status = H5Sclose (memspace);
    status = H5Tclose (Hmemtype);
    free(Hdata);

    // WRITE THE CRC data

    hsize_t crc_dim[1] = {9};
    file_dataspace = H5Screate(H5S_SIMPLE);
    H5Sset_extent_simple(file_dataspace, 1, crc_dim, NULL);

    hid_t dcpl = H5Pcreate (H5P_DATASET_CREATE);
    status = H5Pset_layout (dcpl, H5D_COMPACT);

    dataset = H5Dcreate2(gid, "CRC_id_mask_x_y_z_vx_vy_vz_phi", H5T_NATIVE_ULONG, file_dataspace,
    			 H5P_DEFAULT, dcpl, H5P_DEFAULT);
    H5Sclose (file_dataspace);
    H5Pclose (dcpl);

    t1 = MPI_Wtime();
    if( Rank == 0 ) {
      ret = H5Dwrite(dataset, H5T_NATIVE_ULONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, CRC_values);
    } else {
      mem_dataspace = H5Screate(H5S_NULL);
      ret = H5Dwrite(dataset, H5T_NATIVE_ULONG, mem_dataspace, mem_dataspace, H5P_DEFAULT, NULL);
      H5Sclose (mem_dataspace);
    }
    H5Dclose (dataset);
    
    timer += MPI_Wtime()-t1;
  }
#endif


#ifdef HDF5_HAVE_MULTI_DATASETS
  if (FileIOType == FileIOHDF) {
    hid_t plist_id;
    plist_id = H5Pcreate(H5P_DATASET_XFER);
    //H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
    t1 = MPI_Wtime();
    H5Dwrite_multi(plist_id, Vars.size(), multi_info);
    timer = MPI_Wtime()-t1;
    if(Rank == 0)  {
      H5D_mpio_actual_io_mode_t actual_io_mode;
      H5Pget_mpio_actual_io_mode( plist_id, &actual_io_mode);
      cout << " collective mode " << actual_io_mode << endl; 
    }

    H5Pclose(plist_id);

    for (size_t i = 0; i < Vars.size(); ++i) {
      H5Dclose(multi_info[i].dset_id);
      H5Sclose(multi_info[i].mem_space_id);
      H5Sclose(multi_info[i].dset_space_id);
    }
  }
#endif

  double mean=0;
  double min;
  double max;
  double *rtimers=NULL;
  if(Rank == 0)  {
    rtimers = (double *) malloc(NRanks*sizeof(double));
  }
  MPI_Gather(&timer, 1, MPI_DOUBLE, rtimers, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  if(Rank == 0)  {

    min = rtimers[0];
    max = min;

    for(int n = 1; n < NRanks; n++) {
      if(rtimers[n] > max)
	max=rtimers[n];
      mean += rtimers[n];
      if(rtimers[n] < min)
	min=rtimers[n];
    }
    free(rtimers);
  }

  if (FileIOType == FileIOHDF)
    ret = H5Gclose(gid);

  // Here we want to set the 
  close();
  //MPI_Barrier(Comm);

  double EndTime = MPI_Wtime();
  double TotalTime = EndTime - StartTime;
  double MaxTotalTime;
  MPI_Reduce(&TotalTime, &MaxTotalTime, 1, MPI_DOUBLE, MPI_MAX, 0, Comm);

  if (Rank == 0) {
    // Obtain file size, this is just for benchmarking purpose.We can set the file size to genericIO by using H5Fgetsize.
    hid_t fid = H5Fopen( const_cast<char *>(FileName.c_str()),H5F_ACC_RDONLY,H5P_DEFAULT);
    if(fid <0)
        throw runtime_error( ("Unable to open the file: ") + FileName);
    hsize_t h5_filesize=0;
    if(H5Fget_filesize(fid,&h5_filesize)<0) {
        H5Fclose(fid);
        throw runtime_error( ("Unable to obtain the HDF5 file size: ") + FileName);
    }
    H5Fclose(fid);
#if defined(HDF5_DERV) || defined(HDF5_HAVE_MULTI_DATASETS)        
    printf("WRITE DATA (mean,min,max) = %.4f %.4f %.4f s,  %.4f %.4f %.4f MB/s \n", mean/NRanks, min, max,
	   (double)h5_filesize/(mean/NRanks) / (1024.*1024.), 
	   (double)h5_filesize/min/(1024.*1024.), (double)h5_filesize/max/(1024.*1024.) );
#endif

    double Rate = ((double) h5_filesize) / MaxTotalTime / (1024.*1024.);
    cout << NRanks << " Procs Wrote " << Vars.size() << " variables to " << FileName <<
            " (" << h5_filesize << " bytes) in " << MaxTotalTime << "s: " <<
            Rate << " MB/s" << endl;
  }


  return;
}
// Note: writing errors are not currently recoverable (one rank may fail
// while the others don't).
// Uncomment or comment this line for debugging
//#if 0
template <bool IsBigEndian>
void GenericIO::write() {

//cerr<<"coming to write "<<endl;
 bool use_hdf5 = false;
 const char *EnvStr1 = getenv("GENERICIO_USE_HDF");
 if (EnvStr1 && string(EnvStr1) == "1"){
#ifdef GENERICIO_HAVE_HDF
    use_hdf5 = true;
#endif
 }
 if(true == use_hdf5) 
  write_hdf();
//#else 
 else {
//cerr<<"coming to mpiio "<<endl;
  const char *Magic = IsBigEndian ? MagicBE : MagicLE;

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

  if (FileIOType == FileIOMPI)
    FH.get() = new GenericFileIO_MPI(SplitComm);
  else if (FileIOType == FileIOMPICollective)
    FH.get() = new GenericFileIO_MPICollective(SplitComm);
  else
    FH.get() = new GenericFileIO_POSIX();

  FH.get()->open(LocalFileName);
//cerr<<"before writing data "<<endl;

  uint64_t Offset = RHLocal.Start;
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

    crc64_invert(CRC, CRCLoc);

    if (HasExtraSpace) {
//cerr<<"writing data "<<endl;
      FH.get()->write(Data, WriteSize + CRCSize, Offset, Vars[i].Name + " with CRC");
    } else {
      FH.get()->write(Data, WriteSize, Offset, Vars[i].Name);
      FH.get()->write(CRCLoc, CRCSize, Offset + WriteSize, Vars[i].Name + " CRC");
    }

    if (HasExtraSpace)
       std::copy(CRCSave, CRCSave + CRCSize, CRCLoc);

    Offset += WriteSize + CRCSize;
  }

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
//#endif
}
#endif
//Uncomment or comment this line for debugging
//#endif //if 0
// Still keep the old code in case we need to check. 
// Uncomment or comment this line for debugging
#if 0
// Note: writing errors are not currently recoverable (one rank may fail
// while the others don't).
template <bool IsBigEndian>
void GenericIO::write() {

  const char *Magic = IsBigEndian ? MagicBE : MagicLE;
#ifdef GENERICIO_HAVE_HDF
  hid_t fid, space, dset, attr, filetype, atype, gid, aid, tid1, rid1, rid2, dxpl_id;
  hid_t fapl_id;  // File access templates
  hid_t sid, dataset;
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
  Partition=0;
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

#ifdef HDF5_DERV
  Hdata = (hacc_t *) malloc (NElems * sizeof (hacc_t));

  Hmemtype = H5Tcreate (H5T_COMPOUND, sizeof (hacc_t));
  H5Tinsert (Hmemtype, "id",
		      HOFFSET (hacc_t, id), H5T_NATIVE_LONG);
  H5Tinsert (Hmemtype, "mask", 
		      HOFFSET (hacc_t, mask), H5T_NATIVE_UINT16);
  H5Tinsert (Hmemtype, "x",
		      HOFFSET (hacc_t, x), H5T_NATIVE_FLOAT);
  H5Tinsert (Hmemtype, "y",
		      HOFFSET (hacc_t, y), H5T_NATIVE_FLOAT);
  H5Tinsert (Hmemtype, "z",
		      HOFFSET (hacc_t, z), H5T_NATIVE_FLOAT);
  H5Tinsert (Hmemtype, "vx",
		      HOFFSET (hacc_t, vx), H5T_NATIVE_FLOAT);
  H5Tinsert (Hmemtype, "vy",
		      HOFFSET (hacc_t, vy), H5T_NATIVE_FLOAT);
  H5Tinsert (Hmemtype, "vz",
		      HOFFSET (hacc_t, vz), H5T_NATIVE_FLOAT);
  H5Tinsert (Hmemtype, "phi",
		      HOFFSET (hacc_t, phi), H5T_NATIVE_FLOAT);

#endif

  uint64_t Offsets_glb;
	
  double t1, timer=0.;
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
	// CRC calculation 
	crc send;
	//	cout << Vars[i].Size << Vars[i].Name << endl;
	send.CRC64 = crc64_omp(Data, NElems*Vars[i].Size);
	send.CRC64_size = NElems*Vars[i].Size;
	struct crc_s *rbufv;
	uint64_t *sbufv;

	int          blocklengths[2] = {1,1};
	MPI_Datatype types[2] = {MPI_UINT64_T, MPI_LONG_LONG_INT };
	MPI_Datatype mpi_crc_type;
	MPI_Aint     offsets[2];

	offsets[0] = offsetof(crc, CRC64);
	offsets[1] = offsetof(crc, CRC64_size);

	MPI_Type_create_struct(2, blocklengths, offsets, types, &mpi_crc_type);
	MPI_Type_commit(&mpi_crc_type);

	rbufv = NULL;
	if(Rank == 0) {
	  rbufv = new crc_s [NRanks*sizeof(struct crc_s)];
	}
	sbufv = new uint64_t [NRanks*sizeof(uint64_t)];

	MPI_Gather( &send, 1, mpi_crc_type, rbufv, 1, mpi_crc_type, 0, MPI_COMM_WORLD);
	if(Rank == 0) {
	  uint64_t CRC_sum = 0;
	  sbufv[0] = 0;
	    int k;
	  for (k =0; k<NRanks; k++) {
	    CRC_sum = crc64_combine(CRC_sum, rbufv[k].CRC64, rbufv[k].CRC64_size);
	    // find offsets
	    if(k > 0)
	      sbufv[k] = sbufv[k-1] + rbufv[k-1].CRC64_size/Vars[i].Size;
	    // cout << k << " : " << sbufv[k] << endl;
	  }
	  delete rbufv;
	  CRC = CRC_sum;
	  //  cout << "CRC_sum" << CRC << endl;
	} else
	  CRC = 0;

	uint64_t Offsets;
	MPI_Scatter( sbufv, 1, MPI_UINT64_T, &Offsets, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD); 
	delete sbufv;
	MPI_Type_free(&mpi_crc_type);

#ifdef HDF5_DERV
	hsize_t ii;
	if( Vars[i].Name.compare("id") == 0) {
	  for (ii=0; ii < NElems; ii++)
	    Hdata[ii].id = *((int64_t *)Data + ii);
	  CRC_values[0] = CRC;
	} else if( Vars[i].Name.compare("mask") == 0) {
	  for (ii=0; ii < NElems; ii++)
	    Hdata[ii].mask = *((uint16_t *)Data + ii);
	  CRC_values[1] = CRC;
	} else if( Vars[i].Name.compare("x") == 0) {
	  for (ii=0; ii < NElems; ii++)
	    Hdata[ii].x = *((float *)Data + ii);
	  CRC_values[2] = CRC;
	} else if( Vars[i].Name.compare("y") == 0) {
	  for (ii=0; ii < NElems; ii++)
	    Hdata[ii].y = *((float *)Data + ii);
	  CRC_values[3] = CRC;
	} else if( Vars[i].Name.compare("z") == 0) {
	  for (ii=0; ii < NElems; ii++)
	    Hdata[ii].z = *((float *)Data + ii);
	  CRC_values[4] = CRC;
	} else if( Vars[i].Name.compare("vx") == 0) {
	  for (ii=0; ii < NElems; ii++)
	    Hdata[ii].vx = *((float *)Data + ii);
	  CRC_values[5] = CRC;
	} else if( Vars[i].Name.compare("vy") == 0) {
	  for (ii=0; ii < NElems; ii++)
	    Hdata[ii].vy = *((float *)Data + ii);
	  CRC_values[6] = CRC;
	} else if( Vars[i].Name.compare("vz") == 0) {
	  for (ii=0; ii < NElems; ii++)
	    Hdata[ii].vz = *((float *)Data + ii);
	  CRC_values[7] = CRC;
	} else if( Vars[i].Name.compare("phi") == 0) {
	  for (ii=0; ii < NElems; ii++) 
	    Hdata[ii].phi = *((float *)Data + ii);
	  CRC_values[8] = CRC;
	  
	}
	Offsets_glb = Offsets;

#else
	if( Vars[i].Name.compare("id") == 0) {
	  dtype = H5T_NATIVE_LONG;
	} else if( Vars[i].Name.compare("mask") == 0) {
	  dtype = H5T_NATIVE_UINT16;
	}else {
	  dtype = H5T_NATIVE_FLOAT;
	}
	
	gfio_hdf->write_hdf_internal(Data, WriteSize, Offsets , Vars[i].Name, dtype, NElems, &CRC, gid, TotElem, i);
#endif
      } else 
#endif
	crc64_invert(CRC, CRCLoc);
        t1 = MPI_Wtime();
	FH.get()->write(Data, WriteSize + CRCSize, Offset, Vars[i].Name + " with CRC");
        timer += MPI_Wtime()-t1;
    } else {
      crc64_invert(CRC, CRCLoc);
      t1 = MPI_Wtime();
      FH.get()->write(Data, WriteSize, Offset, Vars[i].Name);
      timer += MPI_Wtime()-t1;
      FH.get()->write(CRCLoc, CRCSize, Offset + WriteSize, Vars[i].Name + " CRC");
    }

    if (HasExtraSpace)
       std::copy(CRCSave, CRCSave + CRCSize, CRCLoc);

    Offset += WriteSize + CRCSize;
  }
#ifdef GENERICIO_HAVE_HDF

#ifdef HDF5_DERV
  if (FileIOType == FileIOHDF) {
   /*
     * Create dataspace.  Setting maximum size to NULL sets the maximum
     * size to be the current size.
     */
    hid_t filespace, memspace, dset, plist_id;
    hsize_t     dims[1];
    dims[0] = (hsize_t)TotElem;
    filespace = H5Screate_simple (1, dims, NULL);

    /*
     * Create the dataset and write the compound data to it.
     */
    dset = H5Dcreate (gid, "DATA", Hmemtype, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    hsize_t count[1];	          /* hyperslab selection parameters */
    hsize_t offset[1];
    herr_t status;
    count[0] = NElems;
    memspace = H5Screate_simple(1, count, NULL);

    offset[0] = Offsets_glb;

    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);

    plist_id = H5Pcreate(H5P_DATASET_XFER);
    //    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

    t1 = MPI_Wtime();
    status = H5Dwrite (dset, Hmemtype, memspace, filespace, plist_id, Hdata);
    timer = MPI_Wtime()-t1;

    H5Pclose(plist_id);
    status = H5Dclose (dset);
    status = H5Sclose (filespace);
    status = H5Sclose (memspace);
    status = H5Tclose (Hmemtype);
    free(Hdata);

    // WRITE THE CRC data

    hsize_t crc_dim[1] = {9};
    file_dataspace = H5Screate(H5S_SIMPLE);
    H5Sset_extent_simple(file_dataspace, 1, crc_dim, NULL);

    hid_t dcpl = H5Pcreate (H5P_DATASET_CREATE);
    status = H5Pset_layout (dcpl, H5D_COMPACT);

    dataset = H5Dcreate2(gid, "CRC_id_mask_x_y_z_vx_vy_vz_phi", H5T_NATIVE_ULONG, file_dataspace,
    			 H5P_DEFAULT, dcpl, H5P_DEFAULT);
    H5Sclose (file_dataspace);
    H5Pclose (dcpl);

    t1 = MPI_Wtime();
    if( Rank == 0 ) {
      ret = H5Dwrite(dataset, H5T_NATIVE_ULONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, CRC_values);
    } else {
      mem_dataspace = H5Screate(H5S_NULL);
      ret = H5Dwrite(dataset, H5T_NATIVE_ULONG, mem_dataspace, mem_dataspace, H5P_DEFAULT, NULL);
      H5Sclose (mem_dataspace);
    }
    H5Dclose (dataset);
    
    timer += MPI_Wtime()-t1;
  }
#endif


#ifdef HDF5_HAVE_MULTI_DATASETS
  if (FileIOType == FileIOHDF) {
    hid_t plist_id;
    plist_id = H5Pcreate(H5P_DATASET_XFER);
    //H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
    t1 = MPI_Wtime();
    H5Dwrite_multi(plist_id, Vars.size(), multi_info);
    timer = MPI_Wtime()-t1;
    if(Rank == 0)  {
      H5D_mpio_actual_io_mode_t actual_io_mode;
      H5Pget_mpio_actual_io_mode( plist_id, &actual_io_mode);
      cout << " collective mode " << actual_io_mode << endl; 
    }

    H5Pclose(plist_id);

    for (size_t i = 0; i < Vars.size(); ++i) {
      H5Dclose(multi_info[i].dset_id);
      H5Sclose(multi_info[i].mem_space_id);
      H5Sclose(multi_info[i].dset_space_id);
    }
  }
#endif
#endif

  double mean=0;
  double min;
  double max;
  double *rtimers=NULL;
  if(Rank == 0)  {
    rtimers = (double *) malloc(NRanks*sizeof(double));
  }
  MPI_Gather(&timer, 1, MPI_DOUBLE, rtimers, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  if(Rank == 0)  {

    min = rtimers[0];
    max = min;

    for(int n = 1; n < NRanks; n++) {
      if(rtimers[n] > max)
	max=rtimers[n];
      mean += rtimers[n];
      if(rtimers[n] < min)
	min=rtimers[n];
    }
    free(rtimers);
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
    printf("WRITE DATA (mean,min,max) = %.4f %.4f %.4f s,  %.4f %.4f %.4f MB/s \n", mean/NRanks, min, max,
	   (double)FileSize/(mean/NRanks) / (1024.*1024.), 
	   (double)FileSize/min/(1024.*1024.), (double)FileSize/max/(1024.*1024.) );
    double Rate = ((double) FileSize) / MaxTotalTime / (1024.*1024.);
    cout << NRanks << " Procs Wrote " << Vars.size() << " variables to " << FileName <<
            " (" << FileSize << " bytes) in " << MaxTotalTime << "s: " <<
            Rate << " MB/s" << endl;
  }

  MPI_Comm_free(&SplitComm);
  SplitComm = MPI_COMM_NULL;
}
// Uncomment or comment this line for debugging
#endif // if 0 or NO MPI

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
void GenericIO::openAndReadHeader_HDF_Simple() {

  int NRanks, Rank;
#ifndef GENERICIO_NO_MPI
  MPI_Comm_rank(Comm, &Rank);
  MPI_Comm_size(Comm, &NRanks);
#else
  Rank = 0;
  NRanks = 1;
#endif

// The following code doesn't make much sense to me to put it here.
// Just follow the origianl openAndReadHeader.
// Doesn't make sense. Still comment out.
#if 0
#ifndef GENERICIO_NO_MPI
  GenericIO GIO(MPI_COMM_SELF, FileName, FileIOType);
#ifdef GENERICIO_HAVE_HDF
  FH.get() = new GenericFileIO_HDF(MPI_COMM_SELF);
#endif
#else
  GenericIO GIO(FileName, FileIOType);
#endif
  FH.get()->open(FileName, true);
#endif

//The following code makes each process to open the file independently.
//For us, we just need to use one process to open the HDF5 file to
//obtain the total number of elements and chunk size. So we just open
//the HDF5 file, peek the information and close it. 

  uint64_t num_elems_array[2]; 
#ifndef GENERICIO_NO_MPI
#ifdef GENERICIO_HAVE_HDF
  if(Rank == 0) {
    hid_t h5file_id = H5Fopen( const_cast<char *>(FileName.c_str()),H5F_ACC_RDONLY,H5P_DEFAULT);
    if(h5file_id <0)
        throw runtime_error( ("Unable to open the file: ") + FileName);
    hid_t h5dset_id  = -1;
    string dsetname;

    // We can do more to find the fit variable name. Here just make it simple and fast.
#ifdef HDF5_DERV 
    dsetname = "/Variables/DATA";
    h5dset_id = H5Dopen2(h5file_id,const_cast<char *>(dsetname.c_str()),H5P_DEFAULT);
#else 
    dsetname = "/Variables/id";
    h5dset_id = H5Dopen2(h5file_id,const_cast<char *>(dsetname.c_str()),H5P_DEFAULT);
#endif
    if(h5dset_id <0) {
      H5Fclose(h5file_id);
      throw runtime_error( ("Unable to open the dataset: ") + dsetname);
    }

    hid_t h5dspace_id = H5Dget_space(h5dset_id);
    hssize_t h5dset_nelems = H5Sget_simple_extent_npoints(h5dspace_id);

    // We can add the retrieval of chunking information later. 
    H5Sclose(h5dspace_id);
    H5Dclose(h5dset_id);
    H5Fclose(h5file_id);

    uint64_t num_elems_0 = (h5dset_nelems/NRanks);
    num_elems_array[0] = h5dset_nelems;

    // Without optimization, just make the number of elements evenly 
    // distributed among processes.
    num_elems_array[1] = (h5dset_nelems/NRanks)+1;
    setTotElem(num_elems_array[0]);
    if((uint64_t)(size_t)num_elems_array[1]!=num_elems_array[1])
        throw runtime_error( "The datatype size_t causes the overflow for number of elements per process. ");
    setNumElems_mine((size_t)num_elems_array[1]);
  }
  MPI_Bcast( num_elems_array, 2, MPI_UINT64_T, 0, Comm);

  if(Rank !=0) {
    setTotElem(num_elems_array[0]);
    if((uint64_t)(size_t)num_elems_array[1]!=num_elems_array[1])
        throw runtime_error( "The datatype size_t causes the overflow for number of elements per process. ");
    setNumElems_mine((size_t)num_elems_array[1]);
  }
#endif

#if 0
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
#endif
#endif

  return;
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
cerr<<"Redistributing "<<endl;
    DisableCollErrChecking = true;

    size_t RowOffset = 0;
cerr<<"SourceRanks.size() is "<<SourceRanks.size() <<endl;
    for (int i = 0, ie = SourceRanks.size(); i != ie; ++i) {
      readData(SourceRanks[i], RowOffset, Rank, TotalReadSize, NErrs);
      RowOffset += readNumElems(SourceRanks[i]);
    }

    DisableCollErrChecking = false;
  } else {
cerr<<"No Redistributing "<<endl;
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
  MPI_Comm_size(Comm, &commRanks);
#else
  Rank = 0;
  commRanks = 1;
#endif

  hid_t dset, dset_id, space, aspace;
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
  uint64_t CRCv[9], CRC_loc;
  hsize_t dim_size[1];
  uint64_t read1_cs = 0;

  hsize_t FileSize;
  double mean=0;
  double min;
  double max;
  double t1, timer;
  double *rtimers=NULL;

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
   // cout << Rank << "  " << dims[0] << endl;

   //sizedims[0] = 1;
    //   mem_dataspace_CRC = H5Screate_simple (1, sizedims, NULL);


#ifdef HDF5_DERV

  if(Rank == 0)  {
    rtimers = (double *) malloc(commRanks*sizeof(double));
  }

   t1 =  MPI_Wtime();
   hid_t memtype;
   dset = H5Dopen(fid, "/Variables/DATA", H5P_DEFAULT);
   dtype = H5Dget_type(dset);
   file_dataspace = H5Dget_space (dset);
   H5Sget_simple_extent_dims(file_dataspace, dim_size, NULL);

   uint64_t *Sendv = NULL;
   if(Rank == 0)
     Sendv = new uint64_t [commRanks*sizeof(uint64_t)];

   MPI_Gather( &dims[0], 1, MPI_UINT64_T, Sendv, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD); // fix comm MSB

   uint64_t *sbufv;
   sbufv = new uint64_t [commRanks*sizeof(uint64_t)];
   
   if(Rank == 0) {
     sbufv[0] = 0;
     int k;
     for (k =0; k<commRanks; k++) {
       // find offsets
       if(k > 0)
	 sbufv[k] = sbufv[k-1] + Sendv[k-1];
       //	 cout << k << " : " << sbufv[k] << endl;
     }
     delete [] Sendv;
   }

   uint64_t Offsets;
   MPI_Scatter( sbufv, 1, MPI_UINT64_T, &Offsets, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD); 
   delete [] sbufv;

   start[0] =  Offsets;
   hypercount[0] = dims[0];
   stride[0] = 1;

   //  cout << start[0] << endl;
   //  cout <<  hypercount[0] << endl;

   ret=H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, start, stride,
			   hypercount, NULL);

   sizedims[0] = dims[0];

   mem_dataspace = H5Screate_simple (1, sizedims, NULL);

   Hdata = (hacc_t *) malloc (sizedims[0] * sizeof (hacc_t));
   
   if (!Hdata) cout << "NULLISH" << endl;

   dxpl_id = H5Pcreate (H5P_DATASET_XFER);
     H5Pset_dxpl_mpio(dxpl_id, H5FD_MPIO_COLLECTIVE);

     //H5Pset_dxpl_checksum_ptr(dxpl_id, &read1_cs);

   memtype = H5Tget_native_type(dtype, H5T_DIR_ASCEND);
   
   FileSize = H5Dget_storage_size(dset);
   ret = H5Dread(dset, dtype, mem_dataspace, file_dataspace, dxpl_id, Hdata);

   H5Pclose(dxpl_id);
   ret = H5Sclose (mem_dataspace);
   ret = H5Sclose (file_dataspace);
   ret = H5Tclose (dtype);
   ret = H5Dclose (dset);
   timer = MPI_Wtime()-t1;

#if 0
   if(Rank == 0) {
     for(size_t j = 0; j < dims[0]; ++j) {
       cout << j << " " << Hdata[j].id << endl;
     }
   }
#endif

   dset_id = H5Dopen(fid, "Variables/CRC_id_mask_x_y_z_vx_vy_vz_phi", H5P_DEFAULT);

   ret = H5Dread(dset_id, H5T_NATIVE_ULONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, &CRCv);
   //  cout << "var size " << Vars[i].Size << endl;

      // CRC calculation 
   crc send[9];
      // send.CRC64_size = dims[0]*Vars[i].Size; // MSB not sure why it does not work

   send[0].CRC64_size = dims[0]*sizeof(long);
   send[1].CRC64_size = dims[0]*sizeof(uint16_t);
   send[2].CRC64_size = dims[0]*sizeof(float);
   send[3].CRC64_size = dims[0]*sizeof(float);
   send[4].CRC64_size = dims[0]*sizeof(float);
   send[5].CRC64_size = dims[0]*sizeof(float);
   send[6].CRC64_size = dims[0]*sizeof(float);
   send[7].CRC64_size = dims[0]*sizeof(float);
   send[8].CRC64_size = dims[0]*sizeof(float);

   void *field;
   hsize_t ii;
   field = (long *)malloc(dims[0]*sizeof(long));
   for (ii=0; ii < dims[0]; ii++)
     *((long *)field + ii) = Hdata[ii].id;
   send[0].CRC64 = crc64_omp(field, send[0].CRC64_size);
   free(field);
   field = (uint16_t *)malloc(dims[0]*sizeof(uint16_t));
   for (ii=0; ii < dims[0]; ii++)
     *((uint16_t *)field + ii) = Hdata[ii].mask;
   send[1].CRC64 = crc64_omp(field, send[1].CRC64_size);
   free(field);
   field = (float *)malloc(dims[0]*sizeof(float));
   for (ii=0; ii < dims[0]; ii++)
     *((float *)field + ii) = Hdata[ii].x;
   send[2].CRC64 = crc64_omp(field, send[2].CRC64_size);
   for (ii=0; ii < dims[0]; ii++)
     *((float *)field + ii) = Hdata[ii].y;
   send[3].CRC64 = crc64_omp(field, send[3].CRC64_size);
   for (ii=0; ii < dims[0]; ii++)
     *((float *)field + ii) = Hdata[ii].z;
   send[4].CRC64 = crc64_omp(field, send[4].CRC64_size);
   for (ii=0; ii < dims[0]; ii++)
     *((float *)field + ii) = Hdata[ii].vx;
   send[5].CRC64 = crc64_omp(field, send[5].CRC64_size);
   for (ii=0; ii < dims[0]; ii++)
     *((float *)field + ii) = Hdata[ii].vy;
   send[6].CRC64 = crc64_omp(field, send[6].CRC64_size);
   for (ii=0; ii < dims[0]; ii++)
     *((float *)field + ii) = Hdata[ii].vz;
   send[7].CRC64 = crc64_omp(field, send[7].CRC64_size);
   for (ii=0; ii < dims[0]; ii++)
     *((float *)field + ii) = Hdata[ii].phi;
   send[8].CRC64 = crc64_omp(field, send[8].CRC64_size);
   free(field);

   struct crc_s *rbufv;
      
   int          blocklengths[2] = {1,1};
   MPI_Datatype types[2] = {MPI_UINT64_T, MPI_LONG_LONG_INT };
   MPI_Datatype mpi_crc_type;
   MPI_Aint     offsets[2];
      
   offsets[0] = offsetof(crc, CRC64);
   offsets[1] = offsetof(crc, CRC64_size);

   MPI_Type_create_struct(2, blocklengths, offsets, types, &mpi_crc_type);
   MPI_Type_commit(&mpi_crc_type);

   rbufv = NULL;
   if(Rank == 0)
     rbufv = new crc_s [9*commRanks*sizeof(struct crc_s)];

   MPI_Gather( &send[0], 9, mpi_crc_type, rbufv, 9, mpi_crc_type, 0, MPI_COMM_WORLD);

   MPI_Type_free(&mpi_crc_type);

   if(Rank == 0) {

     int icnt,ii;
     for (ii=0; ii < 9; ii++) {
       icnt = ii;
       uint64_t CRC_sum = 0;
       for (int k=0; k<commRanks; k++) {
	 CRC_sum = crc64_combine(CRC_sum, rbufv[icnt].CRC64, rbufv[icnt].CRC64_size);
	 icnt += 9 ;
       }
       cout << "Checking CRC for Dataset " << ii ;
       if (CRCv[ii] != CRC_sum || CRC_sum == 0) {
	 cout << " CRC error " << CRCv[ii] << " " << CRC_sum << endl;
       } else {
	 cout << " PASSED" << endl;
       }
     }
   }
   free(rbufv);

   free(Hdata);

   MPI_Gather(&timer, 1, MPI_DOUBLE, rtimers, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   if(Rank == 0)  {
     
     min = rtimers[0];
     max = min;
     mean = min;
     for(int n = 1; n < commRanks; n++) {
       if(rtimers[n] > max)
	 max=rtimers[n];
       mean += rtimers[n];
       if(rtimers[n] < min)
	 min=rtimers[n];
     }
     free(rtimers);
   }
  if (Rank == 0) {
    cout << FileSize << endl;
    printf("%d READ DATA (mean,min,max) = %.4f %.4f %.4f s,  %.4f %.4f %.4f MB/s \n", commRanks, mean/commRanks, min, max,
	   (double)FileSize/(mean/commRanks) / (1024.*1024.), 
	   (double)FileSize/min/(1024.*1024.), (double)FileSize/max/(1024.*1024.) );
  }

  ret = H5Dclose(dset_id);

#else

   mem_dataspace_CRC = H5Screate(H5S_SCALAR);

   for (size_t i = 0; i < Vars.size(); ++i) {

//     uint64_t Offset = RH->Start;
     bool VarFound = false;

     Vars[i].Name = v[i];
     //cout << "var name " << Vars[i].Name << endl;
     //  cout << "var size " << Vars[i].Size << endl;


     std::strcpy(c_str, v[i].c_str());

     // cout << c_str << endl;
     // cout << fid << endl;
     dset = H5Dopen(fid, c_str, H5P_DEFAULT);

     file_dataspace = H5Dget_space (dset);

     H5Sget_simple_extent_dims(file_dataspace, dim_size, NULL);


//      //     FH.get()->read(Data, ReadSize, Offset, Vars[i].Name);
     
     uint64_t *Sendv = NULL;
     if(Rank == 0)
       Sendv = new uint64_t [commRanks*sizeof(uint64_t)];

     MPI_Gather( &dims[0], 1, MPI_UINT64_T, Sendv, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD); // fix comm MSB

     uint64_t *sbufv;
     sbufv = new uint64_t [commRanks*sizeof(uint64_t)];

     if(Rank == 0) {
       sbufv[0] = 0;
       int k;
       for (k =0; k<commRanks; k++) {
	 // find offsets
	 if(k > 0)
	   sbufv[k] = sbufv[k-1] + Sendv[k-1];
	 //	 cout << k << " : " << sbufv[k] << endl;
       }
       delete [] Sendv;
     }

     uint64_t Offsets;
     MPI_Scatter( sbufv, 1, MPI_UINT64_T, &Offsets, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD); 
     delete [] sbufv;

     start[0] =  Offsets;
     hypercount[0] = dims[0];
     stride[0] = 1;

     //  cout << start[0] << endl;
     //  cout <<  hypercount[0] << endl;

     ret=H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, start, stride,
 	    hypercount, NULL);

     sizedims[0] = dims[0];

     mem_dataspace = H5Screate_simple (1, sizedims, NULL);

     void *Data = Vars[i].Data;
     if (!Data) cout << "NULLISH" << endl;

     dxpl_id = H5Pcreate (H5P_DATASET_XFER);
//     H5Pset_dxpl_mpio(dxpl_id, H5FD_MPIO_COLLECTIVE);

     //H5Pset_dxpl_checksum_ptr(dxpl_id, &read1_cs);

     size_t Vsize;

     //   if(Rank==0) printf("Reading Dataset  %s \n",c_str);
     if( Vars[i].Name.compare("/Variables/id") == 0) {
       dtype = H5T_NATIVE_LONG;
       Vsize = sizeof(long);
       Data = new long [dims[0]];
     } else if( Vars[i].Name.compare("/Variables/mask") == 0) {
       dtype = H5T_NATIVE_UINT16;
       Vsize = 2;
       Data = new short int [dims[0]];
     } else  {
       dtype = H5T_NATIVE_FLOAT;
       Vsize = sizeof(float);
       Data = new float [dims[0]];
     }
     ret = H5Dread(dset, dtype, mem_dataspace, file_dataspace, dxpl_id, Data);

#if 0
     if(Rank == 4) {
    if(Vars[i].Name.compare("/Variables/id") == 0) {
      for(size_t j = 0; j < dims[0]; ++j){
	cout << j << " " << ((size_t *)Data)[j] << endl;
      }
    }
     }
#endif
    
    H5Pclose(dxpl_id);
     ret = H5Sclose (mem_dataspace);
     ret = H5Sclose (file_dataspace);
     ret = H5Dclose (dset);

     std::strcpy(c_str3, c_str);
     strcat(c_str3, "_CRC");

     //  cout << "2H5Dread" << ret << endl;
     //    if(Rank==0) printf("Reading in CRC for Dataset %s \n",c_str3);

      //  cout << "2H5Dread" << ret << endl;
#if 0
      attr_id = H5Dopen(fid, c_str3, H5P_DEFAULT);
      file_dataspace_CRC = H5Dget_space(attr_id);

      ret = H5Dread(attr_id, H5T_NATIVE_ULONG, mem_dataspace_CRC, file_dataspace_CRC, H5P_DEFAULT, &CRCv);
      //  cout << "var size " << Vars[i].Size << endl;

      // CRC calculation 
      crc send; 
      // send.CRC64_size = dims[0]*Vars[i].Size; // MSB not sure why it does not work
      send.CRC64_size = dims[0]*Vsize;
      send.CRC64 = crc64_omp(Data, send.CRC64_size);

      struct crc_s *rbufv;
      
      int          blocklengths[2] = {1,1};
      MPI_Datatype types[2] = {MPI_UINT64_T, MPI_LONG_LONG_INT };
      MPI_Datatype mpi_crc_type;
      MPI_Aint     offsets[2];
      
      offsets[0] = offsetof(crc, CRC64);
      offsets[1] = offsetof(crc, CRC64_size);

      MPI_Type_create_struct(2, blocklengths, offsets, types, &mpi_crc_type);
      MPI_Type_commit(&mpi_crc_type);

      rbufv = NULL;
      if(Rank == 0)
	rbufv = new crc_s [commRanks*sizeof(struct crc_s)];

      MPI_Gather( &send, 1, mpi_crc_type, rbufv, 1, mpi_crc_type, 0, MPI_COMM_WORLD);
      
      if(Rank == 0) {
	uint64_t CRC_sum = 0;
	for (int k=0; k<commRanks; k++) {
	  CRC_sum = crc64_combine(CRC_sum, rbufv[k].CRC64, rbufv[k].CRC64_size);
	}
	cout << "Checking CRC for Dataset " << c_str3 ;
	if (CRCv != CRC_sum || CRC_sum == 0) {
	  cout << " CRC error " << CRCv << " " << CRC_sum << endl;
	} else {
	  cout << " PASSED" << endl;
	}
	free(rbufv);
      }
      //     cout << CRCv << endl;

//     if(Vars[i].Name.compare("/Variables/phi") == 0) {
//       for(size_t j = 0; j < dims[0]; ++j){
// 	cout << j << " " << ((float*)Data)[j] << endl;
//       }
//     }
      //   CRC_loc = crc64_omp(Vars[i].Data, dims[0]);

      ret = H5Dclose(attr_id);
      ret = H5Sclose(file_dataspace_CRC);
      
   //    if(Rank==0) printf("Checking in CRC for Dataset %s ",c_str3);
//       if (CRCv[0] != CRC_loc) {
	
//         cout << "CRC error " << CRCv[0] << " " << CRC_loc << endl;
//       } else {
	
//       if(Rank==0) printf("PASSED \n",c_str3);
//       }
      MPI_Type_free(&mpi_crc_type);
#endif
      Data = NULL;
   }
   
   H5Sclose(mem_dataspace_CRC);

#endif
   
   //ret=H5Fclose(fid);
   //MPI_Barrier(MPI_COMM_WORLD);
   // cout << "after close" << endl;

   delete[] c_str;
   delete[] c_str3;

}
#endif

// Implement a simple read module. Only data for both derived and 9 vars.
// Make sure to send the nelems from rank 0 to rank NRank-1 for the Nelems.
// Or we can add an offset element and just scatter 2 elements.KY
#ifdef GENERICIO_HAVE_HDF
  void GenericIO::readDataHDF_Simple() {
  int Rank, commRanks;

  std::vector<std::string> v;
#ifndef GENERICIO_NO_MPI
  MPI_Comm_rank(Comm, &Rank);
  MPI_Comm_size(Comm, &commRanks);
#else
  Rank = 0;
  commRanks = 1;
#endif

  hid_t dset, dset_id, space, aspace;
  hid_t dset_CRC;
  hid_t fid, fapl_id,dtype,dxpl_id;
  herr_t ret;
  hsize_t dims[1]; // dataspace dim sizes
  hid_t dataset, file_dataspace, mem_dataspace;
  hid_t dataset_CRC, file_dataspace_CRC, mem_dataspace_CRC;
  hsize_t start[1];			/* for hyperslab setting */
  hsize_t hypercount[1], stride[1];	/* for hyperslab setting */
  hsize_t sizedims[1];
  uint64_t CRCv[9], CRC_loc;
  hsize_t dim_size[1];
  uint64_t read1_cs = 0;

  hsize_t FileSize;
  double mean=0;
  double min;
  double max;
  double t1, timer;
  double *rtimers=NULL;

#if 0
  std::vector<std::string> v;
  v.push_back("/Variables/id");
  v.push_back("/Variables/mask");
  v.push_back("/Variables/phi");
  v.push_back("/Variables/vx");
  v.push_back("/Variables/vy");
  v.push_back("/Variables/vz");
  v.push_back("/Variables/x");
  v.push_back("/Variables/y");
  v.push_back("/Variables/z");
#endif

//cerr<<"coming to read HDF simple "<<endl;
  fapl_id = H5Pcreate (H5P_FILE_ACCESS);
  //ret = H5Pset_fapl_mpiposix(fapl_id, Comm, 0); 
  
#ifndef GENERICIO_NO_MPI
  ret = H5Pset_fapl_mpio(fapl_id, Comm, MPI_INFO_NULL);
#endif
  if( (fid = H5Fopen(const_cast<char *>(FileName.c_str()),H5F_ACC_RDONLY,fapl_id)) < 0)
      throw runtime_error( ("Unable to open the file: ") + FileName);
  H5Pclose(fapl_id);
#if 0
  GenericFileIO_HDF *gfio_hdf = dynamic_cast<GenericFileIO_HDF *> (FH.get());
  fid = gfio_hdf->get_fileid();
#endif
  //dims[0] = gfio_hdf->get_NumElem();
  dims[0] = (hsize_t)NElems;

  if(Rank == 0)  {
    rtimers = (double *) malloc(commRanks*sizeof(double));
  }

  t1 =  MPI_Wtime();
  start[0] =  Rank*dims[0];
  if(Rank != commRanks-1) 
      hypercount[0] = dims[0];
  else 
      hypercount[0] = TotElem - start[0]; 
  stride[0] = 1;

#ifdef HDF5_DERV

  hid_t memtype;
  dset = H5Dopen(fid, "/Variables/DATA", H5P_DEFAULT);
  dtype = H5Dget_type(dset);
  file_dataspace = H5Dget_space(dset);
   
   // For the simple read case, every rank except the last one will read the
   // same number of elements. So the offset is simple to calculate. 
   // For customized offsets, we should compute/assign them at the
   // openAndReadHeader_HDF_Simple() and distribute them to all processes.

#if 0
   start[0] =  Rank*dims[0];
   if(Rank != NRanks-1) 
      hypercount[0] = dims[0];
   else 
      hypercount[0] = TotElem - start[0]; 
   stride[0] = 1;
#endif

   //  cout << start[0] << endl;
   //  cout <<  hypercount[0] << endl;

   ret=H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, start, stride,
			   hypercount, NULL);

   if(Rank != commRanks-1)
     sizedims[0] = dims[0];
   else
     sizedims[0] = TotElem - Rank*dims[0];


   mem_dataspace = H5Screate_simple (1, sizedims, NULL);

   Hdata = (hacc_t *) malloc (sizedims[0] * sizeof (hacc_t));
   
   if (!Hdata) cout << "NULLISH" << endl;

   dxpl_id = H5Pcreate (H5P_DATASET_XFER);
//     H5Pset_dxpl_mpio(dxpl_id, H5FD_MPIO_COLLECTIVE);

   memtype = H5Tget_native_type(dtype, H5T_DIR_ASCEND);

   //ret = H5Dread(dset, dtype, mem_dataspace, file_dataspace, dxpl_id, Hdata);
   ret = H5Dread(dset, memtype, mem_dataspace, file_dataspace, dxpl_id, Hdata);

   H5Pclose(dxpl_id);
   ret = H5Sclose (mem_dataspace);
   ret = H5Sclose (file_dataspace);
   ret = H5Tclose (memtype);
   ret = H5Tclose (dtype);
   ret = H5Dclose (dset);
   timer = MPI_Wtime()-t1;

#if 0
   if(Rank == 0) {
     for(size_t j = 0; j < sizedims[0]; ++j) {
       cout << j << " " << Hdata[j].id << endl;
     }
   }
#endif
   free(Hdata);


#else

   // Need to build a map between Var name(relative name) later.
   // Now just assume the variable name is consistent.
   // If it is not, H5Dopen will fail.

   for (size_t i = 0; i < Vars.size(); ++i) {

     string var_abo_path = "/Variables/"+Vars[i].Name;
//cerr<<"var path is "<<var_abo_path <<endl;

     dset = H5Dopen(fid, var_abo_path.c_str(), H5P_DEFAULT);

     dtype = H5Dget_type(dset);

     file_dataspace = H5Dget_space (dset);

     //H5Sget_simple_extent_dims(file_dataspace, dim_size, NULL);

     ret=H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, start, stride,
 	    hypercount, NULL);

     if(Rank != commRanks-1)
       sizedims[0] = dims[0];
     else
       sizedims[0] = TotElem - Rank*dims[0];

     mem_dataspace = H5Screate_simple (1, sizedims, NULL);


     dxpl_id = H5Pcreate (H5P_DATASET_XFER);
     // H5Pset_dxpl_mpio(dxpl_id, H5FD_MPIO_COLLECTIVE);
     //H5Pset_dxpl_checksum_ptr(dxpl_id, &read1_cs);

     //size_t Vsize;

     //   if(Rank==0) printf("Reading Dataset  %s \n",c_str);
     hid_t memtype = H5Tget_native_type(dtype,H5T_DIR_ASCEND);
     size_t memtype_size = H5Tget_size(memtype);
     vector <char>Data;
     Data.resize(memtype_size*sizedims[0]);
#if 0
     if( Vars[i].Name.compare("id") == 0){ 
       Data.resize(mem*dims[0]);
     }
     else if( Vars[i].Name.compare("mask") == 0){ 
       memtype = H5T_NATIVE_UINT16;

       Data.resize(2*dims[0]);
       //Data = new unsigned short [dims[0]];
     }
     else {  
       memtype = H5T_NATIVE_FLOAT;
       Data = new float [dims[0]];
     }
#endif
     
     ret = H5Dread(dset, memtype, mem_dataspace, file_dataspace, dxpl_id, (void*)&Data[0]);

#if 0
     if(Rank == 2) {
cerr<<"Rank = "<<Rank <<endl;
    if(Vars[i].Name.compare("id") == 0) {
      for(size_t j = 0; j < dims[0]; ++j){
	    cout << j << " " << *((long *)&Data[0]+j) << endl;
      }
    }
     }

//#if 0
     else if(Rank == 3) {
cerr<<"Rank = "<<Rank <<endl;
    if(Vars[i].Name.compare("id") == 0) {
      for(size_t j = 0; j < sizedims[0]; ++j){
	    cout << j << " " << *((long *)&Data[0]+j) << endl;
      }
    }
     }

#endif
    
     H5Pclose(dxpl_id);
     ret = H5Sclose (mem_dataspace);
     ret = H5Sclose (file_dataspace);
     ret = H5Dclose (dset);
   }
   
#endif
   ret = H5Fclose(fid);

   timer = MPI_Wtime()-t1;
   MPI_Gather(&timer, 1, MPI_DOUBLE, rtimers, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   if(Rank == 0)  {
     
     min = rtimers[0];
     max = min;
     mean = min;
     for(int n = 1; n < commRanks; n++) {
       mean += rtimers[n];
       if(rtimers[n] > max)
	       max=rtimers[n];
       if(rtimers[n] < min)
	       min=rtimers[n];
     }
     free(rtimers);

    // Obtain file size, this is just for benchmarking purpose.We can set the file size to genericIO if necessary.
    hid_t fid = H5Fopen( const_cast<char *>(FileName.c_str()),H5F_ACC_RDONLY,H5P_DEFAULT);
    if(fid <0)
        throw runtime_error( ("Unable to open the file: ") + FileName);
    hsize_t h5_filesize=0;
    if(H5Fget_filesize(fid,&h5_filesize)<0) {
        H5Fclose(fid);
        throw runtime_error( ("Unable to obtain the HDF5 file size: ") + FileName);
    }
    H5Fclose(fid);
    printf("%d READ DATA (mean,min,max) = %.4f %.4f %.4f s,  %.4f %.4f %.4f MB/s \n", commRanks, mean/commRanks, min, max,
	   (double)h5_filesize/(mean/commRanks) / (1024.*1024.), 
	   (double)h5_filesize/min/(1024.*1024.), (double)h5_filesize/max/(1024.*1024.) );
  }  
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

cerr<<"coming to read data "<< "Rank is " << Rank <<endl;
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

cerr<<"before blosc decompression "<<endl;
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
