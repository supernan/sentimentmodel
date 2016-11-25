/*!
 *  \file	SentiClassifierAPI.h
 *  \brief  �����������۵������Է������
 *  \date	2013-8-28
 *  \author ��ѧ��(xuxueke@software.ict.ac.cn)
 */
#ifndef __ICT_SENTI_CLASSIFIER_API_H__
#define __ICT_SENTI_CLASSIFIER_API_H__

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#define ICT_SENTI_CLF_API __declspec(dllexport)
#else
#define ICT_SENTI_CLF_API
#endif

#define MAX_LABEL_LEN 255


    typedef void SENTI_CLASSIFIER_HANDLER;

    /*!
    *	\fn	int Senti_Classifier_Init(const char* config_file, SENTI_CLASSIFIER_HANDLER **h_clf);
    *	\brief �����Է����������ʼ��
    *   \param [in] config_file �����ļ�
    *   \param [out] h_clf �����Է��������
    *	\return	�ɹ����� 1�����򷵻� 0
    *	\author xxk
    *	\date 2013-8-28
    */
    ICT_SENTI_CLF_API int Senti_Classifier_Init(const char *config_file, SENTI_CLASSIFIER_HANDLER **h_clf);


    /*!
    *	\fn	int Senti_Classifier_Exit(CLASSIFIER_HANDLE *h_clf);
    *	\brief ����������ͷ�
    *   \param [in] h_clf �����Է��������
    *	\return	�ɹ����� 1�����򷵻� 0
    *	\author xxk
    *	\date 2013-8-28
    */
    ICT_SENTI_CLF_API int Senti_Classifier_Exit(SENTI_CLASSIFIER_HANDLER *h_clf);



    /*!
    * \fn int Senti_Classify_Perform(SENTI_CLASSIFIER_HANDLER *h_clf, const char* text, double* score);
    * \brief ִ�з���
    * \param [in] text �����ı�
    * \param [in] h_clf �����Է��������
    * \param [out] score �����Է���÷�
    * \return	�ɹ����� 1�����򷵻� 0
    * \date 2013-8-28
    *  \author xxk
    */
    ICT_SENTI_CLF_API int Senti_Classify_Perform(SENTI_CLASSIFIER_HANDLER *h_clf, const char *text, double *score);


#ifdef __cplusplus
}
#endif

#endif

