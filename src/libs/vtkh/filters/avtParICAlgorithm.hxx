#include <iostream>
#include <string.h>
#include "MemStream.h"
#include <vtkh/filters/DebugMeowMeow.hpp>
//#include "avtParICAlgorithm.h"

using namespace std;

avtParICAlgorithm::avtParICAlgorithm(MPI_Comm comm)
{
    MPI_Comm_size(comm, &nProcs);
    MPI_Comm_rank(comm, &rank);
    msgID = 0;
}

void
avtParICAlgorithm::InitializeBuffers(int mSz,
                                     int nMsgRecvs,
                                     int nICRecvs,
                                     int nDSRecvs)
{
    numMsgRecvs = nMsgRecvs;
    numSLRecvs = nICRecvs;
    numDSRecvs = nDSRecvs;

    // Msgs are handled as vector<int>.
    // Serialization of msg consists: size_t (num elements) +
    // sender rank + message size.
    int msgSize = sizeof(size_t);
    msgSize += sizeof(int); // sender rank.
    msgSize += (mSz * sizeof(int));

    //During particle advection, the IC state is only serialized.
    slSize = 256;
    slsPerRecv = 64;

    int dsSize = 2*sizeof(int);

    messageTagInfo[avtParICAlgorithm::MESSAGE_TAG] = pair<int,int>(numMsgRecvs, msgSize);
    messageTagInfo[avtParICAlgorithm::PARTICLE_TAG] = pair<int,int>(numSLRecvs, slSize*slsPerRecv);
    messageTagInfo[avtParICAlgorithm::DATASET_PREP_TAG] = pair<int,int>(numDSRecvs, dsSize);

    //Setup receive buffers.
    map<int, pair<int, int> >::const_iterator it;
    for (it = messageTagInfo.begin(); it != messageTagInfo.end(); it++)
    {
        int tag = it->first, num = it->second.first;
        for (int i = 0; i < num; i++)
            PostRecv(tag);
    }
}

void
avtParICAlgorithm::CleanupRequests(int tag)
{
    vector<RequestTagPair> delKeys;
    for (bufferIterator i = recvBuffers.begin(); i != recvBuffers.end(); i++)
    {
        if (tag == -1 || tag == i->first.second)
            delKeys.push_back(i->first);
    }

    if (! delKeys.empty())
    {
        vector<RequestTagPair>::const_iterator it;
        for (it = delKeys.begin(); it != delKeys.end(); it++)
        {
            RequestTagPair v = *it;

            unsigned char *buff = recvBuffers[v];
            MPI_Cancel(&(v.first));
            delete [] buff;
            recvBuffers.erase(v);
        }
    }
}

void
avtParICAlgorithm::PostRecv(int tag)
{
    map<int, pair<int, int> >::const_iterator it = messageTagInfo.find(tag);
    if (it != messageTagInfo.end())
        PostRecv(tag, it->second.second);
}

void
avtParICAlgorithm::PostRecv(int tag, int sz, int src)
{
    sz += sizeof(avtParICAlgorithm::Header);
    unsigned char *buff = new unsigned char[sz];
    memset(buff, 0, sz);

    MPI_Request req;
    if (src == -1)
        MPI_Irecv(buff, sz, MPI_BYTE, MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &req);
    else
        MPI_Irecv(buff, sz, MPI_BYTE, src, tag, MPI_COMM_WORLD, &req);

    RequestTagPair entry(req, tag);
    recvBuffers[entry] = buff;

    //cerr<<"PostRecv: ("<<req<<", "<<tag<<") buff= "<<(void*)buff<<" sz= "<<sz<<endl;
}

void
avtParICAlgorithm::CheckPendingSendRequests()
{
    bufferIterator it;
    vector<MPI_Request> req, copy;
    vector<int> tags;

    for (it = sendBuffers.begin(); it != sendBuffers.end(); it++)
    {
        req.push_back(it->first.first);
        copy.push_back(it->first.first);
        tags.push_back(it->first.second);
    }

    if (req.empty())
        return;

    //See if any sends are done.
    int num = 0, *indices = new int[req.size()];
    MPI_Status *status = new MPI_Status[req.size()];
    int err = MPI_Testsome(req.size(), &req[0], &num, indices, status);
    if (err != MPI_SUCCESS)
    {
        cerr << "Err with MPI_Testsome in PARIC algorithm" << endl;
    }
    for (int i = 0; i < num; i++)
    {
        MPI_Request r = copy[indices[i]];
        int tag = tags[indices[i]];

        RequestTagPair k(r,tag);
        bufferIterator entry = sendBuffers.find(k);
        if (entry != sendBuffers.end())
        {
            delete [] entry->second;
            sendBuffers.erase(entry);
        }
    }

    delete [] indices;
    delete [] status;
}

bool
avtParICAlgorithm::PacketCompare(const unsigned char *a, const unsigned char *b)
{
    avtParICAlgorithm::Header ha, hb;
    memcpy(&ha, a, sizeof(ha));
    memcpy(&hb, b, sizeof(hb));

    return ha.packet < hb.packet;
}

void
avtParICAlgorithm::PrepareForSend(int tag, MemStream *buff, vector<unsigned char *> &buffList)
{
    map<int, pair<int, int> >::const_iterator it = messageTagInfo.find(tag);
    if (it == messageTagInfo.end())
        throw "message tag not found";

    int bytesLeft = buff->len();
    int maxDataLen = it->second.second;

    avtParICAlgorithm::Header header;
    header.tag = tag;
    header.rank = rank;
    header.id = msgID;
    header.numPackets = 1;
    if (buff->len() > (unsigned int)maxDataLen)
        header.numPackets += buff->len() / maxDataLen;

    header.packet = 0;
    header.packetSz = 0;
    header.dataSz = 0;
    msgID++;

    buffList.resize(header.numPackets);

    size_t pos = 0;
    for (int i = 0; i < header.numPackets; i++)
    {
        header.packet = i;
        if (i == (header.numPackets-1))
            header.dataSz = bytesLeft;
        else
            header.dataSz = maxDataLen;

        header.packetSz = header.dataSz + sizeof(header);
        unsigned char *b = new unsigned char[header.packetSz];

        //Write the header.
        unsigned char *bPtr = b;
        memcpy(bPtr, &header, sizeof(header));
        bPtr += sizeof(header);

        //Write the data.
        memcpy(bPtr, &buff->data()[pos], header.dataSz);
        pos += header.dataSz;

        buffList[i] = b;
        bytesLeft -= maxDataLen;
    }
}

void
avtParICAlgorithm::SendData(int dst, int tag, MemStream *buff)
{
    vector<unsigned char *> bufferList;

    //Add headers, break into multiple buffers if needed.
    PrepareForSend(tag, buff, bufferList);

    avtParICAlgorithm::Header header;
    for (size_t i = 0; i < bufferList.size(); i++)
    {
        memcpy(&header, bufferList[i], sizeof(header));
        MPI_Request req;
        int err = MPI_Isend(bufferList[i], header.packetSz, MPI_BYTE, dst,
                            tag, MPI_COMM_WORLD, &req);
        if (err != MPI_SUCCESS)
        {
            cerr << "Err with MPI_Isend in PARIC algorithm" << endl;
        }
        //BytesCnt.value += header.packetSz;

        //Add it to sendBuffers
        RequestTagPair entry(req, tag);
        sendBuffers[entry] = bufferList[i];
    }

    delete buff;
}

bool
avtParICAlgorithm::RecvData(int tag, std::vector<MemStream *> &buffers,
                            bool blockAndWait)
{
    std::set<int> setTag;
    setTag.insert(tag);
    std::vector<std::pair<int, MemStream *> > b;
    buffers.resize(0);
    if (RecvData(setTag, b, blockAndWait))
    {
        buffers.resize(b.size());
        for (size_t i = 0; i < b.size(); i++)
            buffers[i] = b[i].second;
        return true;
    }
    return false;
}

bool
avtParICAlgorithm::RecvData(set<int> &tags,
                            vector<pair<int, MemStream *> > &buffers,
                            bool blockAndWait)
{
    buffers.resize(0);

    //Find all recv of type tag.
    vector<MPI_Request> req, copy;
    vector<int> reqTags;
    for (bufferIterator i = recvBuffers.begin(); i != recvBuffers.end(); i++)
    {
        if (tags.find(i->first.second) != tags.end())
        {
            req.push_back(i->first.first);
            copy.push_back(i->first.first);
            reqTags.push_back(i->first.second);
        }
    }

    if (req.empty())
        return false;

    MPI_Status *status = new MPI_Status[req.size()];
    int *indices = new int[req.size()], num = 0;
    if (blockAndWait)
        MPI_Waitsome(req.size(), &req[0], &num, indices, status);
    else
        MPI_Testsome(req.size(), &req[0], &num, indices, status);

    if (num == 0)
    {
        delete [] status;
        delete [] indices;
        return false;
    }

    vector<unsigned char *> incomingBuffers(num);
    for (int i = 0; i < num; i++)
    {
        RequestTagPair entry(copy[indices[i]], reqTags[indices[i]]);
        bufferIterator it = recvBuffers.find(entry);
        if ( it == recvBuffers.end())
        {
            delete [] status;
            delete [] indices;
            throw "receive buffer not found";
        }

        incomingBuffers[i] = it->second;
        recvBuffers.erase(it);
    }

    ProcessReceivedBuffers(incomingBuffers, buffers);

    for (int i = 0; i < num; i++)
        PostRecv(reqTags[indices[i]]);

    delete [] status;
    delete [] indices;

    return ! buffers.empty();
}

void
avtParICAlgorithm::ProcessReceivedBuffers(vector<unsigned char*> &incomingBuffers,
                                          vector<pair<int, MemStream *> > &buffers)
{
    for (size_t i = 0; i < incomingBuffers.size(); i++)
    {
        unsigned char *buff = incomingBuffers[i];

        //Grab the header.
        avtParICAlgorithm::Header header;
        memcpy(&header, buff, sizeof(header));

        //Only 1 packet, strip off header and add to list.
        if (header.numPackets == 1)
        {
            MemStream *b = new MemStream(header.dataSz, (buff + sizeof(header)));
            b->rewind();
            pair<int, MemStream*> entry(header.tag, b);
            buffers.push_back(entry);
            delete [] buff;
        }

        //Multi packet....
        else
        {
            RankIdPair k(header.rank, header.id);
            packetIterator i2 = recvPackets.find(k);

            //First packet. Create a new list and add it.
            if (i2 == recvPackets.end())
            {
                list<unsigned char *> l;
                l.push_back(buff);
                recvPackets[k] = l;
            }
            else
            {
                i2->second.push_back(buff);

                // The last packet came in, merge into one MemStream.
                if (i2->second.size() == (size_t)header.numPackets)
                {
                    //Sort the packets into proper order.
                    i2->second.sort(avtParICAlgorithm::PacketCompare);

                    MemStream *mergedBuff = new MemStream;
                    list<unsigned char *>::iterator listIt;

                    for (listIt = i2->second.begin(); listIt != i2->second.end(); listIt++)
                    {
                        unsigned char *bi = *listIt;

                        avtParICAlgorithm::Header header;
                        memcpy(&header, bi, sizeof(header));
                        mergedBuff->write(&bi[sizeof(header)], header.dataSz);
                        delete [] bi;
                    }

                    mergedBuff->rewind();
                    pair<int, MemStream*> entry(header.tag, mergedBuff);
                    buffers.push_back(entry);
                    recvPackets.erase(i2);
                }
            }
        }
    }
}

void
avtParICAlgorithm::SendMsg(int dst, vector<int> &msg)
{
    MemStream *buff = new MemStream;

    //Write data.
    buff->write(rank);
    buff->write(msg);

    SendData(dst, avtParICAlgorithm::MESSAGE_TAG, buff);
//    MsgCnt.value++;
//    CommTime.value += visitTimer->StopTimer(timerHandle, "SendMsg");
}

void
avtParICAlgorithm::SendAllMsg(vector<int> &msg)
{
    for (int i = 0; i < nProcs; i++)
        if (i != rank)
        {
            DBG("          ***************** SendMsg to "<<i<<" "<<msg<<endl);
            SendMsg(i, msg);
        }
}

bool
avtParICAlgorithm::RecvAny(vector<MsgCommData> *msgs,
                           list<ParticleCommData<Particle>> *recvICs,
                           vector<DSCommData> *ds,
                           bool blockAndWait)
{
    set<int> tags;
    if (msgs)
    {
        tags.insert(avtParICAlgorithm::MESSAGE_TAG);
        msgs->resize(0);
    }
    if (recvICs)
    {
        tags.insert(avtParICAlgorithm::PARTICLE_TAG);
        recvICs->resize(0);
    }
    if (ds)
    {
        tags.insert(avtParICAlgorithm::DATASET_TAG);
        tags.insert(avtParICAlgorithm::DATASET_PREP_TAG);
        ds->resize(0);
    }

    if (tags.empty())
        return false;

    vector<pair<int, MemStream *> > buffers;
    if (! RecvData(tags, buffers, blockAndWait))
        return false;

//    int timerHandle = visitTimer->StartTimer();

    for (size_t i = 0; i < buffers.size(); i++)
    {
        if (buffers[i].first == avtParICAlgorithm::MESSAGE_TAG)
        {
            int sendRank;
            vector<int> m;
            buffers[i].second->read(sendRank);
            buffers[i].second->read(m);
            MsgCommData msg(sendRank, m);

            msgs->push_back(msg);
        }
        else if (buffers[i].first == avtParICAlgorithm::PARTICLE_TAG)
        {
            int num, sendRank;
            buffers[i].second->read(sendRank);
            buffers[i].second->read(num);
            for (int j = 0; j < num; j++)
            {
                Particle recvP;
                buffers[i].second->read(recvP);
                ParticleCommData<Particle> d(sendRank, recvP);
                recvICs->push_back(d);
            }
        }
        else if (buffers[i].first == avtParICAlgorithm::DATASET_TAG)
        {
            /*
            BlockIDType dom;
            buffers[i].second->read(dom);

            vtkDataSet *d;
            buffers[i].second->read(&d);
            DSCommData dsData(dom, d);
            ds->push_back(dsData);
            */
        }
        else if (buffers[i].first == avtParICAlgorithm::DATASET_PREP_TAG)
        {
            int sendRank, dsLen;
            buffers[i].second->read(sendRank);
            buffers[i].second->read(dsLen);

            PostRecv(avtParICAlgorithm::DATASET_TAG, dsLen);
        }

        delete buffers[i].second;
    }

//    CommTime.value += visitTimer->StopTimer(timerHandle, "RecvAny");
    return true;
}

bool
avtParICAlgorithm::RecvMsg(vector<MsgCommData> &msgs)
{
    return RecvAny(&msgs, NULL, NULL, false);
}

template <typename P, template <typename, typename> class Container,
          typename Allocator>
void avtParICAlgorithm::SendICs(int dst, Container<P, Allocator> &c)
{
    if (dst == rank)
    {
        cerr<<"Error. Sending IC to yourself"<<endl;
        return;
    }
    if (c.empty())
        return;

    MemStream *buff = new MemStream;
    buff->write(rank);
    int num = c.size();
    buff->write(num);
    for (auto &p : c)
        buff->write(p);
    SendData(dst, avtParICAlgorithm::PARTICLE_TAG, buff);
    c.clear();
}

template <typename P, template <typename, typename> class Container,
          typename Allocator>
void avtParICAlgorithm::SendICs(std::map<int, Container<P, Allocator>> &m)
{
    for (auto mit = m.begin(); mit != m.end(); mit++)
        if (! mit->second.empty())
            SendICs(mit->first, mit->second);
}

template <typename P, template <typename, typename> class Container,
          typename Allocator>
bool avtParICAlgorithm::RecvICs(Container<ParticleCommData<P>, Allocator> &recvICs)
{
    return RecvAny(NULL, &recvICs, NULL, false);
}

template <typename P, template <typename, typename> class Container,
          typename Allocator>
bool avtParICAlgorithm::RecvICs(Container<P, Allocator> &recvICs)
{
    list<ParticleCommData<P>> incoming;

    if (RecvICs(incoming))
    {
        for (auto &it : incoming)
            recvICs.push_back(it.p);
        return true;
    }
    return false;

/*

    bool val = RecvICs(incoming);
    if (val)
    {
        list<ParticleCommData<Particle>>::iterator it;
        for (it = incoming.begin(); it != incoming.end(); it++)
            recvICs.push_back((*it).p);
    }

    return val;
*/
}

template <typename P>
bool avtParICAlgorithm::DoSendICs(int dst, vector<P> &ics)
{
    if (dst == rank)
    {
        cerr << "Error in avtParICAlgorithm::DoSendICs() Sending ICs to yourself" << endl;
        for (size_t i = 0; i < ics.size(); i++)
            cerr << "Proc " << rank << "  "<<ics[i]<<endl;
    }

    if (ics.empty())
        return false;

    MemStream *buff = new MemStream;
    buff->write(rank);
    int num = ics.size();
    buff->write(num);
    //cout<<" ********************* DOSENDICS: "<<ics[0]<<endl;
    for (size_t i = 0; i < ics.size(); i++)
        buff->write(ics[i]);
    SendData(dst, avtParICAlgorithm::PARTICLE_TAG, buff);

    return true;
}



#if 0
void
avtParICAlgorithm::SendDS(int dst, vector<vtkDataSet *> &ds, vector<BlockIDType> &doms)
{
    int timerHandle = visitTimer->StartTimer();

    //Serialize the data sets.
    for (size_t i = 0; i < ds.size(); i++)
    {
        MemStream *dsBuff = new MemStream;

        dsBuff->write(doms[i]);
        dsBuff->write(ds[i]);
        int totalLen = dsBuff->len();

        MemStream *msgBuff = new MemStream(2*sizeof(int));
        msgBuff->write(rank);
        msgBuff->write(totalLen);
        SendData(dst, avtParICAlgorithm::DATASET_PREP_TAG, msgBuff);
        MsgCnt.value++;

        //Send dataset.
        messageTagInfo[avtParICAlgorithm::DATASET_TAG] = pair<int,int>(1, totalLen+sizeof(avtParICAlgorithm::Header));
        SendData(dst, avtParICAlgorithm::DATASET_TAG, dsBuff);
        messageTagInfo.erase(messageTagInfo.find(avtParICAlgorithm::DATASET_TAG));

        DSCnt.value++;
    }
    CommTime.value += visitTimer->StopTimer(timerHandle, "SendDS");
}
#endif

#if 0
bool
avtParICAlgorithm::RecvDS(vector<DSCommData> &ds)
{
    return RecvAny(NULL, NULL, &ds, false);
}
#endif
