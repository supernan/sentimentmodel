#ifndef __ICT_TEXT_CLASSIFIER_API_H__
#define __ICT_TEXT_CLASSIFIER_API_H__

#ifdef _WIN32
#define ICT_TEXT_CLF_API __declspec(dllexport)
#else
#define ICT_TEXT_CLF_API
#endif

#include <string>
using namespace std;

typedef void CLASSIFIER_HANDLER;

/*!
*	\brief 分类器句柄初始化
*   \param [in] config_file 配置文件
*   \param [out] h_clf 分类器句柄
*	\return	成功返回 1，否则返回 0
*	\author xxk
*	\date 2013-8-28
*/
ICT_TEXT_CLF_API int Classifier_Init(const string &config_file, CLASSIFIER_HANDLER *&h_clf);



/*!
*	\fn	int Classifier_Exit(CLASSIFIER_HANDLER *h_clf);
*	\brief 分类器句柄释放
*   \param [in] h_clf 分类器句柄
*	\return	成功返回 1，否则返回 0
*	\author xxk
*	\date 2013-8-28
*/
ICT_TEXT_CLF_API int Classifier_Exit(CLASSIFIER_HANDLER *h_clf);


/*!
* \brief 执行分类
* \param [in] text 输入文本
* \param [in] h_clf 分类器句柄
* \param [out] label 得分最大的类别名称
* \param [out] score 分类得分, 如果分类得分低于给定阈值, 可以认为与任何类别都无关, 具体阈值待定
* \return	成功返回 1，否则返回 0
* \date 2013-8-28
*  \author xxk
*/
ICT_TEXT_CLF_API int Text_Classify_Perform_ZX(CLASSIFIER_HANDLER *h_clf, const string &text, string &label, double &score);


#endif

