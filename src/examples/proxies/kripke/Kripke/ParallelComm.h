/* Common declarations for the functions in comm.c */
#ifndef KRIPKE_COMM_H__
#define KRIPKE_COMM_H__

#include<vector>
#include<mpi.h>

class Grid_Data;
class Subdomain;

class ParallelComm {
  public:
    explicit ParallelComm(Grid_Data *grid_data_ptr);
    virtual ~ParallelComm();
    
    // Adds a subdomain to the work queue
    virtual void addSubdomain(int sdom_id, Subdomain &sdom) = 0;

    // Checks if there are any outstanding subdomains to complete
    // false indicates all work is done, and all sends have completed
    virtual bool workRemaining(void);

    // Returns a vector of ready subdomains, and clears them from the ready queue
    virtual std::vector<int> readySubdomains(void) = 0;

    // Marks subdomains as complete, and performs downwind communication
    virtual void markComplete(int sdom_id) = 0;

    static int getIncomingRequests();
    static int getOutgoingRequests();
    static void resetRequests();
    
  protected:
    static int computeTag(int mpi_rank, int sdom_id);
    static void computeRankSdom(int tag, int &mpi_rank, int &sdom_id);
    int findSubdomain(int sdom_id);
    Subdomain *dequeueSubdomain(int sdom_id);
    void postRecvs(int sdom_id, Subdomain &sdom);
    void postSends(Subdomain *sdom, double *buffers[3]);
    void testRecieves(void);
    void waitAllSends(void);
    std::vector<int> getReadyList(void);


    Grid_Data *grid_data;

    // These vectors contian the recieve requests
    std::vector<MPI_Request> recv_requests;
    std::vector<int> recv_subdomains;

    // These vectors have the subdomains, and the remaining dependencies
    std::vector<int> queue_sdom_ids;
    std::vector<Subdomain *> queue_subdomains;
    std::vector<int> queue_depends;

    // These vectors have the remaining send requests that are incomplete
    std::vector<MPI_Request> send_requests;
};


class SweepComm : public ParallelComm {
  public:
    explicit SweepComm(Grid_Data *data);
    virtual ~SweepComm();

    virtual void addSubdomain(int sdom_id, Subdomain &sdom);
    virtual bool workRemaining(void);
    virtual std::vector<int> readySubdomains(void);
    virtual void markComplete(int sdom_id);
};


class BlockJacobiComm : public ParallelComm {
  public:
    explicit BlockJacobiComm(Grid_Data *data);
    virtual ~BlockJacobiComm();

    void addSubdomain(int sdom_id, Subdomain &sdom);
    bool workRemaining(void);
    std::vector<int> readySubdomains(void);
    void markComplete(int sdom_id);

  private:
    bool posted_sends;
};



#endif
