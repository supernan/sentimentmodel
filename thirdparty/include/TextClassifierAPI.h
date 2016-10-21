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
*	\brief �����������ʼ��
*   \param [in] config_file �����ļ�
*   \param [out] h_clf ���������
*	\return	�ɹ����� 1�����򷵻� 0
*	\author xxk
*	\date 2013-8-28
*/
ICT_TEXT_CLF_API int Classifier_Init(const string &config_file, CLASSIFIER_HANDLER *&h_clf);



/*!
*	\fn	int Classifier_Exit(CLASSIFIER_HANDLER *h_clf);
*	\brief ����������ͷ�
*   \param [in] h_clf ���������
*	\return	�ɹ����� 1�����򷵻� 0
*	\author xxk
*	\date 2013-8-28
*/
ICT_TEXT_CLF_API int Classifier_Exit(CLASSIFIER_HANDLER *h_clf);


/*!
* \brief ִ�з���
* \param [in] text �����ı�
* \param [in] h_clf ���������
* \param [out] label �÷������������
* \param [out] score ����÷�, �������÷ֵ��ڸ�����ֵ, ������Ϊ���κ�����޹�, ������ֵ����
* \return	�ɹ����� 1�����򷵻� 0
* \date 2013-8-28
*  \author xxk
*/
ICT_TEXT_CLF_API int Text_Classify_Perform_ZX(CLASSIFIER_HANDLER *h_clf, const string &text, string &label, double &score);


#endif

