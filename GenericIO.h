#ifndef GENERICIO_H
#define GENERICIO_H

#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>
#include <limits>
#include <stdint.h>

#ifndef GENERICIO_NO_MPI
#include <mpi.h>
#else
#include <fstream>
#endif

#include <unistd.h>

namespace gio {

class GenericFileIO {
public:
  virtual ~GenericFileIO() {}

public:
  virtual void open(const std::string &FN, bool ForReading = false) = 0;
  virtual void setSize(size_t sz) = 0;
  virtual void read(void *buf, size_t count, off_t offset,
                    const std::string &D) = 0;
  virtual void write(const void *buf, size_t count, off_t offset,
                     const std::string &D) = 0;

protected:
  std::string FileName;
};

#ifndef GENERICIO_NO_MPI
class GenericFileIO_MPI : public GenericFileIO {
public:
  GenericFileIO_MPI(const MPI_Comm &C) : FH(MPI_FILE_NULL), Comm(C) {}
  virtual ~GenericFileIO_MPI();

public:
  virtual void open(const std::string &FN, bool ForReading = false);
  virtual void setSize(size_t sz);
  virtual void read(void *buf, size_t count, off_t offset, const std::string &D);
  virtual void write(const void *buf, size_t count, off_t offset, const std::string &D);

protected:
  MPI_File FH;
  MPI_Comm Comm;
};

class GenericFileIO_MPICollective : public GenericFileIO_MPI {
public:
  GenericFileIO_MPICollective(const MPI_Comm &C) : GenericFileIO_MPI(C) {}

public:
  void read(void *buf, size_t count, off_t offset, const std::string &D);
  void write(const void *buf, size_t count, off_t offset, const std::string &D);
};
#endif

class GenericFileIO_POSIX : public GenericFileIO {
public:
  GenericFileIO_POSIX() : FH(-1) {}
  ~GenericFileIO_POSIX();

public:
  void open(const std::string &FN, bool ForReading = false);
  void setSize(size_t sz);
  void read(void *buf, size_t count, off_t offset, const std::string &D);
  void write(const void *buf, size_t count, off_t offset, const std::string &D);

protected:
  int FH;
};

class GenericIO {
public:
  enum VariableFlags {
    VarHasExtraSpace =  (1 << 0), // Note that this flag indicates that the
                                  // extra space is available, but the GenericIO
                                  // implementation is required to
                                  // preserve its contents.
    VarIsPhysCoordX  =  (1 << 1),
    VarIsPhysCoordY  =  (1 << 2),
    VarIsPhysCoordZ  =  (1 << 3),
    VarMaybePhysGhost = (1 << 4)
  };

  struct VariableInfo {
    VariableInfo(const std::string &N, std::size_t S, bool IF, bool IS,
                 bool PCX, bool PCY, bool PCZ, bool PG)
      : Name(N), Size(S), IsFloat(IF), IsSigned(IS),
        IsPhysCoordX(PCX), IsPhysCoordY(PCY), IsPhysCoordZ(PCZ),
        MaybePhysGhost(PG) {}

    std::string Name;
    std::size_t Size;
    bool IsFloat;
    bool IsSigned;
    bool IsPhysCoordX, IsPhysCoordY, IsPhysCoordZ;
    bool MaybePhysGhost;
  };

public:
  struct Variable {
    template <typename T>
    Variable(const std::string &N, T* D, unsigned Flags = 0)
      : Name(N), Size(sizeof(T)),
        IsFloat(!std::numeric_limits<T>::is_integer),
        IsSigned(std::numeric_limits<T>::is_signed),
        Data((void *) D), HasExtraSpace(Flags & VarHasExtraSpace),
        IsPhysCoordX(Flags & VarIsPhysCoordX),
        IsPhysCoordY(Flags & VarIsPhysCoordY),
        IsPhysCoordZ(Flags & VarIsPhysCoordZ),
        MaybePhysGhost(Flags & VarMaybePhysGhost) {}

    Variable(const VariableInfo &VI, void *D, unsigned Flags = 0)
      : Name(VI.Name), Size(VI.Size), IsFloat(VI.IsFloat),
        IsSigned(VI.IsSigned), Data(D),
        HasExtraSpace(Flags & VarHasExtraSpace),
        IsPhysCoordX((Flags & VarIsPhysCoordX) || VI.IsPhysCoordX),
        IsPhysCoordY((Flags & VarIsPhysCoordY) || VI.IsPhysCoordY),
        IsPhysCoordZ((Flags & VarIsPhysCoordZ) || VI.IsPhysCoordZ),
        MaybePhysGhost((Flags & VarMaybePhysGhost) || VI.MaybePhysGhost) {}

    std::string Name;
    std::size_t Size;
    bool IsFloat;
    bool IsSigned;
    void *Data;
    bool HasExtraSpace;
    bool IsPhysCoordX, IsPhysCoordY, IsPhysCoordZ;
    bool MaybePhysGhost;
  };

public:
  enum FileIO {
    FileIOMPI,
    FileIOPOSIX,
    FileIOMPICollective
  };

#ifndef GENERICIO_NO_MPI
  GenericIO(const MPI_Comm &C, const std::string &FN, unsigned FIOT = -1)
    : NElems(0), FileIOType(FIOT == (unsigned) -1 ? DefaultFileIOType : FIOT),
      Partition(DefaultPartition), Comm(C), FileName(FN), SplitComm(MPI_COMM_NULL) {
    std::fill(PhysOrigin, PhysOrigin + 3, 0.0);
    std::fill(PhysScale,  PhysScale + 3, 0.0);
  }
#else
  GenericIO(const std::string &FN, unsigned FIOT = -1)
    : NElems(0), FileIOType(FIOT == (unsigned) -1 ? DefaultFileIOType : FIOT),
      Partition(DefaultPartition), FileName(FN) {
    std::fill(PhysOrigin, PhysOrigin + 3, 0.0);
    std::fill(PhysScale,  PhysScale + 3, 0.0);
  }
#endif

  ~GenericIO() {
    close();

#ifndef GENERICIO_NO_MPI
    if (SplitComm != MPI_COMM_NULL)
      MPI_Comm_free(&SplitComm);
#endif
  }

public:
  std::size_t requestedExtraSpace() const {
    return 8;
  }

  void setNumElems(std::size_t E) {
    NElems = E;

#ifndef GENERICIO_NO_MPI
    int IsLarge = E >= CollectiveMPIIOThreshold;
    int AllIsLarge;
    MPI_Allreduce(&IsLarge, &AllIsLarge, 1, MPI_INT, MPI_SUM, Comm);
    if (!AllIsLarge)
      FileIOType = FileIOMPICollective;
#endif
  }

  void setPhysOrigin(double O, int Dim = -1) {
    if (Dim >= 0)
      PhysOrigin[Dim] = O;
    else
      std::fill(PhysOrigin, PhysOrigin + 3, O);
  }

  void setPhysScale(double S, int Dim = -1) {
    if (Dim >= 0)
      PhysScale[Dim] = S;
    else
      std::fill(PhysScale,  PhysScale + 3, S);
  }

  template <typename T>
  void addVariable(const std::string &Name, T *Data,
                   unsigned Flags = 0) {
    Vars.push_back(Variable(Name, Data, Flags));
  }

  template <typename T, typename A>
  void addVariable(const std::string &Name, std::vector<T, A> &Data,
                   unsigned Flags = 0) {
    T *D = Data.empty() ? 0 : &Data[0];
    addVariable(Name, D, Flags);
  }

  void addVariable(const VariableInfo &VI, void *Data,
                   unsigned Flags = 0) {
    Vars.push_back(Variable(VI, Data, Flags));
  }

#ifndef GENERICIO_NO_MPI
  // Writing
  void write();
#endif

  // Reading
  void openAndReadHeader(bool MustMatch = true, int EffRank = -1,
                         bool CheckPartMap = true);

  int readNRanks();
  void readDims(int Dims[3]);

  // Note: For partitioned inputs, this returns -1.
  uint64_t readTotalNumElems();

  void readPhysOrigin(double Origin[3]);
  void readPhysScale(double Scale[3]);

  void clearVariables() { this->Vars.clear(); };

  int getNumberOfVariables() { return this->Vars.size(); };


  void getVariableInfo(std::vector<VariableInfo> &VI);

  std::size_t readNumElems(int EffRank = -1);
  void readCoords(int Coords[3], int EffRank = -1);
  int readGlobalRankNumber(int EffRank = -1);

 void readData(int EffRank = -1, bool PrintStats = true, bool CollStats = true);

  void close() {
    FH.close();
  }

  void setPartition(int P) {
    Partition = P;
  }

  static void setDefaultFileIOType(unsigned FIOT) {
    DefaultFileIOType = FIOT;
  }

  static void setDefaultPartition(int P) {
    DefaultPartition = P;
  }

  static void setNaturalDefaultPartition();

  static void setDefaultShouldCompress(bool C) {
    DefaultShouldCompress = C;
  }

#ifndef GENERICIO_NO_MPI
  static void setCollectiveMPIIOThreshold(std::size_t T) {
#ifndef GENERICIO_NO_NEVER_USE_COLLECTIVE_IO
    CollectiveMPIIOThreshold = T;
#endif
  }
#endif

protected:
  std::vector<Variable> Vars;
  std::size_t NElems;

  double PhysOrigin[3], PhysScale[3];

  unsigned FileIOType;
  int Partition;
#ifndef GENERICIO_NO_MPI
  MPI_Comm Comm;
#endif
  std::string FileName;

  static unsigned DefaultFileIOType;
  static int DefaultPartition;
  static bool DefaultShouldCompress;

#ifndef GENERICIO_NO_MPI
  static std::size_t CollectiveMPIIOThreshold;
#endif

  std::vector<int> RankMap;
#ifndef GENERICIO_NO_MPI
  MPI_Comm SplitComm;
#endif
  std::string OpenFileName;

  // This reference counting mechanism allows the the GenericIO class
  // to be used in a cursor mode. To do this, make a copy of the class
  // after reading the header but prior to adding the variables.
  struct FHManager {
    FHManager() : CountedFH(0) {
      allocate();
    }

    FHManager(const FHManager& F) {
      CountedFH = F.CountedFH;
      CountedFH->Cnt += 1;
    }

    ~FHManager() {
      close();
    }

    GenericFileIO *&get() {
      if (!CountedFH)
        allocate();

      return CountedFH->GFIO;
    }

    std::vector<char> &getHeaderCache() {
      if (!CountedFH)
        allocate();

      return CountedFH->HeaderCache;
    }

    void allocate() {
      close();
      CountedFH = new FHWCnt;
    };

    void close() {
      if (CountedFH && CountedFH->Cnt == 1)
        delete CountedFH;
      else if (CountedFH)
        CountedFH->Cnt -= 1;

      CountedFH = 0;
    }

    struct FHWCnt {
      FHWCnt() : GFIO(0), Cnt(1) {}

      ~FHWCnt() {
        close();
      }

protected:
      void close() {
        delete GFIO;
        GFIO = 0;
      }

public:
      GenericFileIO *GFIO;
      size_t Cnt;

      // Used for reading
      std::vector<char> HeaderCache;
    };

    FHWCnt *CountedFH;
  } FH;
};

} /* END namespace cosmotk */
#endif // GENERICIO_H

