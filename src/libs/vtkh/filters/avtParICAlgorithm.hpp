#ifndef AVT_PAR_IC_ALGORITHM_H
#define AVT_PAR_IC_ALGORITHM_H

#include <mpi.h>
#include <list>
#include <vector>
#include <set>
#include <map>
#include "CommData.hpp"
#include "Particle.hpp"

class MemStream;

class avtParICAlgorithm
{
  public:
    avtParICAlgorithm(MPI_Comm comm);
    ~avtParICAlgorithm() {}

    void InitializeBuffers(int msgSize,
                           int numMsgRecvs,
                           int numICRecvs,
                           int numDSRecvs=0);
    void Cleanup() { CleanupRequests(); }


    //Manage communication.
    void CleanupRequests(int tag=-1);
    void CheckPendingSendRequests();

    // Send/Recv Integral curves.
    template <typename P, template <typename, typename> class Container,
              typename Allocator=std::allocator<P>>
    void SendICs(int dst, Container<P, Allocator> &c);

    template <typename P, template <typename, typename> class Container,
              typename Allocator=std::allocator<P>>
    void SendICs(std::map<int, Container<P, Allocator>> &m);

    template <typename P, template <typename, typename> class Container,
              typename Allocator=std::allocator<P>>
    bool RecvICs(Container<P, Allocator> &recvICs);

    template <typename P, template <typename, typename> class Container,
              typename Allocator=std::allocator<P>>
    bool RecvICs(Container<ParticleCommData<P>, Allocator> &recvICs);

    // Send/Recv messages.
    void SendMsg(int dst, std::vector<int> &msg);
    void SendAllMsg(std::vector<int> &msg);
    bool RecvMsg(std::vector<MsgCommData> &msgs);

    // Send/Recv datasets.
//  void SendDS(int dst, std::vector<vtkDataSet *> &ds, std::vector<BlockIDType> &doms);
//  bool RecvDS(std::vector<DSCommData> &ds);
    bool RecvAny(std::vector<MsgCommData> *msgs,
                 std::list<ParticleCommData<Particle>> *recvICs,
                 std::vector<DSCommData> *ds,
                 bool blockAndWait);

  private:
    void PostRecv(int tag);
    void PostRecv(int tag, int sz, int src=-1);
    void SendData(int dst, int tag, MemStream *buff);
    bool RecvData(std::set<int> &tags,
                  std::vector<std::pair<int,MemStream *>> &buffers,
                  bool blockAndWait=false);
    bool RecvData(int tag, std::vector<MemStream *> &buffers,
                  bool blockAndWait=false);
    void AddHeader(MemStream *buff);
    void RemoveHeader(MemStream *input, MemStream *header, MemStream *buff);

    template <typename P>
    bool DoSendICs(int dst, std::vector<P> &ics);
    void PrepareForSend(int tag, MemStream *buff, std::vector<unsigned char *> &buffList);
    static bool PacketCompare(const unsigned char *a, const unsigned char *b);
    void ProcessReceivedBuffers(std::vector<unsigned char*> &incomingBuffers,
                                std::vector<std::pair<int, MemStream *>> &buffers);

    // Send/Recv buffer management structures.
    typedef std::pair<MPI_Request, int> RequestTagPair;
    typedef std::pair<int, int> RankIdPair;
    typedef std::map<RequestTagPair, unsigned char *>::iterator bufferIterator;
    typedef std::map<RankIdPair, std::list<unsigned char *>>::iterator packetIterator;

    int rank, nProcs;

    std::map<RequestTagPair, unsigned char *> sendBuffers, recvBuffers;
    std::map<RankIdPair, std::list<unsigned char *>> recvPackets;

    // Maps MPI_TAG to pair(num buffers, data size).
    std::map<int, std::pair<int, int>> messageTagInfo;
    int numMsgRecvs, numSLRecvs, numDSRecvs;
    int slSize, slsPerRecv, msgSize;
    long msgID;

    enum
    {
        MESSAGE_TAG = 42000,
        PARTICLE_TAG = 42001,
        DATASET_PREP_TAG = 42002,
        DATASET_TAG = 42003
    };

    //Message headers.
    typedef struct
    {
        int rank, id, tag, numPackets, packet, packetSz, dataSz;
    } Header;
};

#include "avtParICAlgorithm.hxx"

#endif
