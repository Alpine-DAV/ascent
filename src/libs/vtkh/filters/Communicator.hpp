#ifndef VTK_H_COMMUNICATOR_HPP
#define VTK_H_COMMUNICATOR_HPP

#ifdef VTKH_PARALLEL

#include <mpi.h>
#include <vtkh/filters/Particle.hpp>
#include <vtkh/filters/avtParICAlgorithm.hpp>
#include <vtkh/filters/BoundsMap.hpp>

#include <vtkh/filters/DebugMeowMeow.hpp>

class MPICommunicator
{
    const int MSG_TERMINATE = 1;
    const int MSG_DONE = 2;

public:
    MPICommunicator(MPI_Comm _mpiComm, const vector<int> &_blockToRank) :
        mpiComm(_mpiComm), a(_mpiComm), blockToRank(_blockToRank)
    {
        MPI_Comm_rank(mpiComm, &rank);
        MPI_Comm_size(mpiComm, &nProcs);

        a.InitializeBuffers(2, nProcs, nProcs);
        termCounter = 0;
        lastTerm = 0;
        done = false;

        int R = 8;
        termSendMap.resize(nProcs);
        if (nProcs <= R)
            for (int i = 0; i < nProcs; i++) termSendMap[i] = 0;
        else
        {
            int S = 2;
            for (int i = 0; i < nProcs; i++)
            {
                if (i < R)
                    termSendMap[i] = 0;
                else
                    termSendMap[i] = i/S;
            }
        }
    }

    virtual void SendDone()
    {
        vector<int> msg = {MSG_DONE};
        a.SendAllMsg(msg);
    }

    virtual size_t Exchange(bool haveWork,
                            list<Particle> &outData,
                            list<Particle> &inData,
                            list<Particle> &term,
                            const BoundsMap &boundsMap,
//                            const Bounds &globalBounds,
//                            reader::BaseReader *reader,
                            int numTerm)
    //                            stats::statisticsDB *sdb)
    {
        DBG(" ---Exchange: O="<<outData.size()<<" I="<<inData.size()<<" NT= "<<numTerm<<endl);

        map<int, list<Particle>> sendData;

        int earlyTerm = 0;
        if (!outData.empty())
        {
            vector<int> blockIds;
            boundsMap.FindBlockIDs(outData, blockIds);
            DBG("          -O.blockIds: "<<outData<<" "<<blockIds<<endl);

            auto bit = blockIds.begin();
            for (auto lit = outData.begin(); lit != outData.end(); lit++, bit++)
            {
                int id = *bit;
                lit->blockId = id;
                if (id == -1)
                {
                    term.push_back(*lit);
                    earlyTerm++;
                    DBG("    earlyterm: "<<*lit<<" id= "<<id<<endl);
                }
                else
                {
                    int dst = blockToRank[id];
                    if (dst == rank)
                        inData.push_back(*lit);
                    else
                        sendData[dst].push_back(*lit);
                }
            }

            DBG("   ---SendP: "<<sendData<<endl);
            for (auto &i : sendData)
                a.SendICs(i.first, i.second);
//            sdb->increment("numMsgSent", sendData.size());
        }
//        if (earlyTerm > 0)
//            sdb->increment("earlyTerm", earlyTerm);


        int terminations = earlyTerm + numTerm;

        if (terminations > 0)
        {
            vector<int> msg = {MSG_TERMINATE, terminations};
            DBG("   ---SendM: rank="<<termSendMap[rank]<<" msg="<<msg<<endl);
            a.SendAllMsg(msg);
        }

        //Check if we have anything coming in.
        vector<MsgCommData> msgData;
        list<ParticleCommData<Particle>> particleData;
        int incomingTerm = 0;

        bool blockAndWait = false;
        DBG(" ---RecvAny..."<<endl);
        if (a.RecvAny(&msgData, &particleData, NULL, blockAndWait))
        {
            DBG(" ---Recv: M: "<<msgData<<" P: "<<particleData<<endl);
//            sdb->increment("numMsgRecv", msgData.size());

            for (auto &m : msgData)
            {
                if (m.message[0] == MSG_TERMINATE)
                {
                    incomingTerm += m.message[1];
                    //numTermReceived += m.message[1];
                    DBG("     TERMinate: Recv: "<<m.message[1]<<endl);
                }
                else if (m.message[0] == MSG_DONE)
                    done = true;
            }
            for (auto &p : particleData)
                inData.push_back(p.p);
/*
            if (numTermReceived > 0)
            {
                if (rank == 0)
                    termCounter += numTermReceived;
                else
                {
                    vector<int> msg(2);
                    msg[0] = MSG_TERMINATE;
                    msg[1] = numTermReceived;
                    a.SendMsg(termSendMap[rank], msg);
//                    sdb->increment("numTermSent");
//                    sdb->increment("numMsgSent");
                }
            }
*/
        }
        else
            DBG("  ---RecvAny --Nothing in the can"<<endl);


        a.CheckPendingSendRequests();
        int returnTerm = incomingTerm + earlyTerm;
        DBG(" ---ExchangeDone: nt= "<<returnTerm<<" I= "<<inData.size()<<" T= "<<term.size()<<endl);
        return returnTerm;

        //old code with some funkiness in it...
#if 0

        //Local terminations.
        numTerm += earlyTerm;
        int termDiff = numTerm - lastTerm;

        if (!haveWork && numTerm != lastTerm)
        {
            if (0)//rank == 0)
            {
                termCounter += termDiff;
//                DBG("     TERMinate: Send to self. termDiff= "<<termDiff<<" termCounter= "<<termCounter<<endl);
            }
            else
            {
                vector<int> msg(2);
                msg[0] = MSG_TERMINATE;
                msg[1] = termDiff;
                DBG("   ---SendM: rank="<<termSendMap[rank]<<" msg="<<msg<<endl);
                a.SendMsg(termSendMap[rank], msg);

//                sdb->increment("numTermSent");
//                sdb->increment("numMsgSent");
                DBG("     TERMinate: Send to 0. termDiff= "<<termDiff<<" numTerm= "<<numTerm<<endl);
            }

            lastTerm = numTerm;
        }
        else if (numTerm != lastTerm)
            DBG("     TERMinate: But have work. numTerm= "<<numTerm<<" termCounter= "<<termCounter<<endl);

        vector<MsgCommData> msgData;
        list<ParticleCommData<Particle>> particleData;

        bool blockAndWait = false;
        if (a.RecvAny(&msgData, &particleData, NULL, blockAndWait))
        {
            DBG(" ---Recv: M: "<<msgData<<" P: "<<particleData<<endl);
//            sdb->increment("numMsgRecv", msgData.size());
            int numTermReceived = 0;
            for (auto &m : msgData)
            {
                if (m.message[0] == MSG_TERMINATE)
                {
                    //termCounter += m.message[1];
                    numTermReceived += m.message[1];
                    DBG("     TERMinate: Recv. termCounter= "<<numTermReceived<<endl);
                }
                else if (m.message[0] == MSG_DONE)
                    done = true;
            }
            for (auto &p : particleData)
                inData.push_back(p.p);
            if (numTermReceived > 0)
            {
                if (rank == 0)
                    termCounter += numTermReceived;
                else
                {
                    vector<int> msg(2);
                    msg[0] = MSG_TERMINATE;
                    msg[1] = numTermReceived;
                    a.SendMsg(termSendMap[rank], msg);
//                    sdb->increment("numTermSent");
//                    sdb->increment("numMsgSent");
                }
            }
        }

        a.CheckPendingSendRequests();
        DBG(" ---ExchangeDone: nt= "<<termCounter<<" I= "<<inData.size()<<endl);
        return termCounter;
#endif
    }

    //map<int,vector<int>> rankToBlock;
    vector<int> blockToRank;

    avtParICAlgorithm a;
    size_t termCounter, lastTerm;
    vector<int> termSendMap;
    MPI_Comm mpiComm;
    int rank, nProcs;
    bool done;
};

#endif

#endif
