#include <Kripke/ParallelComm.h>
#include <Kripke/SubTVec.h>
#include <Kripke/Grid.h>

#include <fcntl.h>
#include <unistd.h>
#include <mpi.h>
#include <vector>
#include <stdio.h>


SweepComm::SweepComm(Grid_Data *data) : ParallelComm(data)
{

}

SweepComm::~SweepComm(){
}

/**
  Adds a subdomain to the work queue.
  Determines if upwind dependencies require communication, and posts appropirate Irecv's.
*/
void SweepComm::addSubdomain(int sdom_id, Subdomain &sdom){
  // Post recieves for upwind dependencies, and add to the queue
  postRecvs(sdom_id, sdom);
}


// Checks if there are any outstanding subdomains to complete
// false indicates all work is done, and all sends have completed
bool SweepComm::workRemaining(void){
  // If there are outstanding subdomains to process, return true
  if(ParallelComm::workRemaining()){
    return true;
  }

  // No more work, so make sure all of our sends have completed
  // before we continue
  waitAllSends();

  return false;
}


/**
  Checks for incomming messages, and returns a list of ready subdomain id's
*/
std::vector<int> SweepComm::readySubdomains(void){
  // check for incomming messages
  testRecieves();

  // build up a list of ready subdomains
  return getReadyList();
}


void SweepComm::markComplete(int sdom_id){
  // Get subdomain pointer and remove from work queue
  Subdomain *sdom = dequeueSubdomain(sdom_id);

  // Send new downwind info for sweep
  double *buf[3] = {
    sdom->plane_data[0]->ptr(),
    sdom->plane_data[1]->ptr(),
    sdom->plane_data[2]->ptr()
  };
  postSends(sdom, buf);
}

