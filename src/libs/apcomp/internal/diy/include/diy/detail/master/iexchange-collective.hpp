namespace apcompdiy
{
    struct Master::IExchangeInfoCollective: public IExchangeInfo
    {
      using IExchangeInfo::IExchangeInfo;

      inline bool       all_done() override;                    // get global all done status
      inline void       add_work(int work) override;            // add work to global work counter
      inline void       control() override;

      int               local_work_ = 0;
      int               dirty = 0;
      int               local_dirty, all_dirty;

      int               state = 0;
      mpi::request      r;

      // debug
      bool              first_ibarrier = true;

      using IExchangeInfo::prof;
    };
}

bool
apcompdiy::Master::IExchangeInfoCollective::
all_done()
{
    return state == 3;
}

void
apcompdiy::Master::IExchangeInfoCollective::
add_work(int work)
{
    local_work_ += work;
    if (local_work_ > 0)
        dirty = 1;
}

void
apcompdiy::Master::IExchangeInfoCollective::
control()
{
    if (state == 0 && local_work_ == 0)
    {
        // debug
        if (first_ibarrier)
        {
            prof << "consensus-time";
            first_ibarrier = false;
        }

        r = ibarrier(comm);
        dirty = 0;
        state = 1;
    } else if (state == 1)
    {
        mpi::optional<mpi::status> ostatus = r.test();
        if (ostatus)
        {
            local_dirty = dirty;
            r = mpi::iall_reduce(comm, local_dirty, all_dirty, std::logical_or<int>());
            state = 2;
        }
    } else if (state == 2)
    {
        mpi::optional<mpi::status> ostatus = r.test();
        if (ostatus)
        {
            if (all_dirty == 0)     // done
                state = 3;
            else
                state = 0;          // reset
        }
    }
}

