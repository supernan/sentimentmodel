#include <iostream>
#include <fstream>
#include "TextClassifierAPI.h"
#include "sentimodel.h"


using namespace senti_analysis;
using namespace std;

static void test()
{
	CLASSIFIER_HANDLER *h_clf;

	if (0 == Classifier_Init("conf/text_classifier.config.xml", h_clf))
	{
		cout<<"Failed to  Init"<<endl;
		return;
	}

       while (true)
       {
		fstream fInFile;
		fInFile.open("test.txt", ios::in);
		string line;
		string text;
		while (getline(fInFile,line))
		{
			text += line;
			text += "\n";
		}

		string label_id;
		double score = 0.0;

		if (1 == Text_Classify_Perform_ZX(h_clf, text, label_id, score))
			cout<<label_id<<" "<<score<<endl;

		fInFile.close();
		char c;
		cin>>c;
       }

	Classifier_Exit(h_clf);
}

int fn_iInitWeiboDataFromFile(const char *dataPath, vector<Weibo> &weibo)
{
		//check path
	if(NULL == dataPath)
	{
		cerr<<"weibo data path error"<<endl;
		return 1;
	}

		//open file
	ifstream in(dataPath);
	if(!in)
		return 1;

		//parse data
	string record;
	int counter = 0;
	while(getline(in, record))
	{
		if(record == "")
			continue;
		Weibo new_doc;
		new_doc.id = counter;
		counter++;
		new_doc.source = record;
		weibo.push_back(new_doc);
	}
	in.close();
	return 0;
}

int main(int argc, char **argv)
{
	//test();
    if (argc < 2)
    {
        cout << "miss input file" << endl;
        return 1;
    }
    string confPath = "../conf/text_classifier.config.xml";
    CSentimentModel model(confPath);
    vector<Weibo> text;
    vector<vector<pstWeibo> > res;
    string fpath = argv[1];
    fn_iInitWeiboDataFromFile(fpath.c_str(), text);
	vector<pstWeibo> corpus;
    for (int i = 0; i < text.size(); i++)
        corpus.push_back(&text[i]);

    vector<double> scores;
    vector<string> labels;
    if (!model.BatchAnalysis(corpus, scores, labels))
    {
        cout<<"sentiment analysis failed" << endl;
        return 1;
    }

    for (int i = 0; i < corpus.size(); i++)
    {
        cout<<i<<" "<<labels[i]<<" "<<scores[i]<<endl;
    }
	return 0;
}

