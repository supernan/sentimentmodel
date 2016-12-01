#include "sentimodel.h"

using namespace WeiboTopic_ICT;

int GetSentiCoreNum()
{
    int count = 0;
	count = sysconf(_SC_NPROCESSORS_CONF);
    return count;
}


void senti_analysis::Lock(CSentimentModel *pSentiModel)
{
    pthread_mutex_lock(&(pSentiModel->m_iModelMutex));
}


void senti_analysis::UnLock(CSentimentModel *pSentiModel)
{
	pthread_mutex_unlock(&(pSentiModel->m_iModelMutex));
}


void* senti_analysis::SentiAnaysisThreadFunc(void *arg)
{
    sentiAnaysisThreadParam *pParam = (sentiAnaysisThreadParam*)arg;
    vector<pstWeibo> *pDocs = pParam->m_pCorpus;
    vector<double> *pScores = pParam->m_pSentiScores;
    int nStart = pParam->m_nStart;
    int nEnd = pParam->m_nEnd;
    CSentimentModel *pSentiModel = pParam->m_hSentiModel;

    map<int, double> mTempScore;

    for (int i = nStart; i < nEnd; i++)
    {
        double score = 0.0;
        if (!pSentiModel->AnalysisDocument((*pDocs)[i], score))
        {
            LOG(WARNING) << "Sentiment analysis failed in thread function" << endl;
            mTempScore[i] = ERROR_SCORE;
            continue;
        }
        else
        {
            mTempScore[i] = score;
        }
    }

    Lock(pSentiModel);
    for (int i = nStart; i < nEnd; i++)
    {
        if (i >= pScores->size())
        {
            LOG(WARNING) << "Doc index is out of boundry in thread function" << i << endl;
            continue;
        }
        (*pScores)[i] = mTempScore[i];
    }
    UnLock(pSentiModel);
}


senti_analysis::CSentimentModel::CSentimentModel(const string &rConfPath)
{
    //FLAGS_log_dir = "../logs/";
    //google::InitGoogleLogging("sentimodel");
    pthread_mutex_init(&m_iModelMutex, NULL);
    if (Senti_Classifier_Init(rConfPath.c_str(), &m_hSentiHandler) != 1)
    {
        LOG(FATAL) << "Sentiment Model init Failed" << endl;
    }
    LOG(INFO) << "SentiModel init succeed" << endl;
}


senti_analysis::CSentimentModel::~CSentimentModel()
{
    pthread_mutex_destroy(&m_iModelMutex);
    Senti_Classifier_Exit(m_hSentiHandler);
    LOG(INFO) << "SentiModel exit" << endl;
}


bool senti_analysis::CSentimentModel::AnalysisDocument(pstWeibo pDoc, double &dScore)
{
    string text = pDoc->source;
    //cout<<"before "<<text<<endl;
    if (Senti_Classify_Perform(m_hSentiHandler, text.c_str(), &dScore) != 1)
    {
        LOG(WARNING) << "Doc Sentiment analysis failed " << text << endl;
        return false;
    }
    return true;
}


bool senti_analysis::CSentimentModel::BatchAnalysis(vector<pstWeibo> &rCorpus,
                                                    vector<double> &vScores)
{
    vScores.clear();
    if (rCorpus.empty())
    {
        LOG(ERROR) << "Corpus empty when BatchAnalysis" << endl;
        return false;
    }

    int nSize = rCorpus.size();
    vScores.assign(nSize, ERROR_SCORE);
    int nCores = GetSentiCoreNum();
    int nThreads = 0;
    if (nCores < 2)
        nThreads = 1;
    else
        nThreads = nCores; //TODO
    int nPatchSize = nSize / nThreads;
    if (nPatchSize <= 0) //if the number of docs is too small
    {
        nThreads = 1;
        nPatchSize = rCorpus.size();
    }

    THPOOL::CThreadPool iSentiThreadPool(nThreads);
	iSentiThreadPool.fn_iInitThread();
    vector<sentiAnaysisThreadParam*> vParams;
    for (int i = 0; i < nThreads; i++)
    {
        sentiAnaysisThreadParam *param = new sentiAnaysisThreadParam;
        if (param == NULL)
        {
            LOG(ERROR) << "Memory allocate failed in BatchAnalysis" << endl;
            return false;
        }
        param->m_pCorpus = &rCorpus;
        param->m_pSentiScores = &vScores;
        param->m_hSentiModel = this;
        param->m_nStart = i * nPatchSize;
        if (i == nThreads - 1)
            param->m_nEnd = rCorpus.size();
        else
            param->m_nEnd = (i+1) * nPatchSize;
        vParams.push_back(param);
		iSentiThreadPool.thpool_add_work(SentiAnaysisThreadFunc, (void*)param);
    }

	iSentiThreadPool.thpool_wait_work();
    for (int i = 0; i < vParams.size(); i++)
    {
        delete vParams[i];
        vParams[i] = NULL;
    }
    LOG(INFO) << "BatchAnalysis Succeed" << endl;
    return true;

}
