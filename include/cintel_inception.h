

typedef struct _Cintel_Inception_Parallelization_Params Cintel_Inception_Parallelization_Params;

class Cintel_Inception
{
public:
    Cintel_Inception(void) { }

    ~Cintel_Inception(void)
    {
        global_session_00.reset();
        global_session_01.reset();
        global_session_02.reset();
        global_session_10.reset();
        global_session_11.reset();
        global_session_12.reset();
    }


private:
    int nthreads;
    std::shared_ptr<tensorflow::Session> global_session_00;
    std::shared_ptr<tensorflow::Session> global_session_01;
    std::shared_ptr<tensorflow::Session> global_session_02;

    std::shared_ptr<tensorflow::Session> global_session_10;
    std::shared_ptr<tensorflow::Session> global_session_11;
    std::shared_ptr<tensorflow::Session> global_session_12;



    //int Parse_Graph(string graph_filename);
    
};

struct _Cintel_Inception_Parallelization_Params
{
    int nthreads;
    int threadid;
    Cintel_Inception *instance;
};

