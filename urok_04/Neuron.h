#pragma once

#include "txtReader.h"

vector<double> vInLayer; //������ ������� ��������
vector<double> vOutLayer; //������ �������� ��������
vector<vector<double>> vHiddenLayer; //��������� ������ �������� �������� ����

vector<vector<double>> vWeightInLayer; //��������� ������ ����� �������� ����
vector<vector<double>> vWeightHiddenLayer; //���������� ������ ����� ������� �����
vector<vector<double>> vWeightOutLayer; //��������� ������ ����� ��������� ����

vector<double> vMultiLayer; //������������ ������� �������� �� ������� ����
vector<double> vSumLayer; //������ ����� ������������
vector<double> vActLayer; //������ ��������� 
vector<double> vAllActLayer; //������ ���� ���������

vector<double> vConcretValues; //������ ��������� ��������
vector<double> vErrLayer; //������ ������
vector<double> vDeltaWeight; //������ ������� �����
double learningRate = 0.5; //����������� ��������
vector<double> vMultiErr; //������ ������������ ����� ����� � ����� �����
vector<double> vSumErr; //������ ���� ������������

vector<double> vTrainingLayer; //������ ������� �������� ��� ��������
vector<double> vTrainingWeight; //������ ������� ����� ��� ��������
vector<double> vTrainingActLayer; //������ �������������� �������� ��� ��������
vector<double> vTrainingBufWeight; //������ ������ �������� ��� ��������

class Neuron
{
	int sizeInLayer; //���������� ������� ��������
	int numberHiddenLayer; //���������� ������� �����
	int minusFirstNumberHiddenLayer;//���������� ������� ����� ��� ����� �������� ����
	int sizeHiddenLayer; //���������� �������� ������ �������� ����
	int sizeOutLayer; //���������� �������� ��������
	int sizeWeightInLayer; //���������� ������� �����
	int sizeWeightHiddenLayer; //���������� ����� �������� ����
	int sizeWeightOutLayer; //���������� �������� �����
	int sizeAllWeight; //���������� ���� �����
	int iterationTraining; //������ ������� �������� ��������
	int iWeightHiddenMin; //�������� �� ���������� ������� �������� ����
	int iWeightHiddenMax; //����������� �� ������������� �������� ��� iWeightHidden
	int s; //������� �������� ��������

	double correctAnsver = 0; //������� ���������� ������� ��� ���������� ��������
	double epochMSE = 1; //������� ���� �� 1 �� 100, 101 = 1
	int counterMSE = 0; //������� ������ ��� ��������

public:

	//����������� ��������� ���������� ������� ��������, ������� �����, �������� ������� �����, �������� ��������
	Neuron(int sInLayer, int nHiddenLayer, int sHiddenLayer, int sOutLayer)
	{
		sizeInLayer = sInLayer;
		numberHiddenLayer = nHiddenLayer;
		minusFirstNumberHiddenLayer = nHiddenLayer - 1;
		sizeHiddenLayer = sHiddenLayer;
		sizeOutLayer = sOutLayer;

		//��������� ����� �������� ���� �� 0 �� sInLayer
		sizeWeightInLayer = sizeInLayer * sizeHiddenLayer;
		//��������� ����� �������� ���� �� sInLayer �� sHiddenLayer
		sizeWeightHiddenLayer = minusFirstNumberHiddenLayer * (sizeHiddenLayer * sizeHiddenLayer);
		//��������� ����� ��������� ���� �� sHiddenLayer �� sOutLayer
		sizeWeightOutLayer = sizeHiddenLayer * sizeOutLayer;

		sizeAllWeight = sizeInLayer * sizeHiddenLayer + minusFirstNumberHiddenLayer * (sizeHiddenLayer * sizeHiddenLayer) + sizeHiddenLayer * sizeOutLayer;

		s = numberHiddenLayer; //�������� �������� ����� = ���������� ������� ����

		cout << "�������� ���������� ��������� ������ �������!" << endl << endl;
	}

	//�-� ��������
	double sigmoid(double x)
	{
		double s;
		s = 1 / (1 + exp(-x));
		return s;
	}
	//�-� ����������� �������
	double sigmoidDx(double x)
	{
		double sDx;
		sDx = sigmoid(x) * (1 - sigmoid(x));
		return sDx;
	}

	//������ ��������� ��������
	void createConcretValues(vector<double> a)
	{
		vConcretValues.resize(sizeOutLayer);

		cout << "��������� ��������: ";
		for (int i = 0; i < vConcretValues.size(); i++)
		{
			vConcretValues[i] = a[i];
			cout << vConcretValues[i] << " ";
		}
		//cout << endl <<"�������� �������� ��������� �������� ������ �������!" << endl << endl;
	}

	//������� ������ ������� ��������
	void createInLayer(vector<double> a)
	{
		vInLayer.resize(sizeInLayer);

		for (int i = 0; i < vInLayer.size(); i++)
		{
			vInLayer[i] = a[i];
		}
/*
		for (int i = 0; i < vInLayer.size(); i++)
		{
			cout << vInLayer[i] << " ";
		}
		cout << endl << "�������� ����������� ������� ������� �������� ������ �������!" << endl << endl;*/
	}

	//������� ������ �������� ��������
	void createOutLayer()
	{
		vOutLayer.resize(sizeOutLayer);

		for (int i = 0; i < vOutLayer.size(); i++)
		{
			vOutLayer[i] = 0.1;
		}

		//for (int i = 0; i < vOutLayer.size(); i++)
		//{
		//	cout << vOutLayer[i] << " ";
		//}
		//cout << endl << "�������� ����������� ������� �������� �������� ������ �������!" << endl << endl;
	}

	//������� ��������� ������ ������� ����� � �������� � ���
	void createHiddenLayer()
	{
		vHiddenLayer.assign(numberHiddenLayer, vector<double>(sizeHiddenLayer)); //(������(�������))

		//������ - ���������� ������� �����
		for (int i = 0; i < vHiddenLayer.size(); i++)
		{
			//������� - ���������� �������� � ����� ������� ����
			for (int j = 0; j < vHiddenLayer[i].size(); j++)
			{
				vHiddenLayer[i][j] = 0.1;
			}
		}

		//for (int i = 0; i < vHiddenLayer.size(); i++)
		//{
		//	for (int j = 0; j < vHiddenLayer[i].size(); j++)
		//	{
		//		cout << vHiddenLayer[i][j] << " ";
		//	}
		//	cout << endl;
		//}
		//cout << endl << "�������� ����������� ������� ������� ����� ������ �������!" << endl << endl;
	}

	//������� ��������� ����� ����� �������� ����
	void createWeightInLayer()
	{
		vWeightInLayer.assign(sizeInLayer, vector<double>(sizeHiddenLayer, 0.5)); //(������(�������))

		//������ - ���������� ������� ��������
		for (int i = 0; i < vWeightInLayer.size(); i++)
		{
			//������� - ���������� ������� �����
			for (int j = 0; j < vWeightInLayer[i].size(); j++)
			{
				vWeightInLayer[i][j] = 0.4;
			}
		}

		//for (int i = 0; i < vWeightInLayer.size(); i++)
		//{
		//	for (int j = 0; j < vWeightInLayer[i].size(); j++)
		//	{
		//		cout << vWeightInLayer[i][j] << " ";
		//	}
		//	cout << endl;
		//}

		//cout << "���������� ����� �������� ����: " << sizeWeightInLayer << endl << endl;
	}

	//������� ���������� ������ ������� �������� � ��������� ����������� ����������
	void createWeightHiddenLayer()
	{
		//vWeightHiddenLayer.assign(minusFirstNumberHiddenLayer, vector<vector<double>>(sizeHiddenLayer, vector<double>(sizeHiddenLayer)));

		vWeightHiddenLayer.assign(sizeHiddenLayer*minusFirstNumberHiddenLayer, vector<double>(sizeHiddenLayer));

		//������ - ���������� �������� ����� �������� ���� � ������� ������� �����
		//for (int i = 0; i < minusFirstNumberHiddenLayer; i++)
		//{
			//������ - ���������� �������� � ����� ������� ����
		for (int j = 0; j < sizeHiddenLayer*minusFirstNumberHiddenLayer; j++)
		{
			//������� - ���������� �������� � ������ ������� ����
			for (int k = 0; k < sizeHiddenLayer; k++)
			{
				vWeightHiddenLayer[j][k] = 0.5;
			}
		}
		//}

	//	//for (int i = 0; i < minusFirstNumberHiddenLayer; i++)
	//	//{
	//		//������ - ���������� �������� � ����� ������� ����
	//	for (int j = 0; j < vWeightHiddenLayer.size(); j++)
	//	{
	//		//������� - ���������� �������� � ������ ������� ����
	//		for (int k = 0; k < vWeightHiddenLayer[j].size(); k++)
	//		{
	//			cout << vWeightHiddenLayer[j][k] << " ";
	//		}
	//		cout << endl;
	//	}
	//	//cout << endl;
	////}
	//	cout << "���������� ����� �������� ����: " << sizeWeightHiddenLayer << endl << endl;
	}

	//������� ��������� ������ ����� �������� ��������
	void createWeightOutLayer()
	{
		vWeightOutLayer.assign(sizeHiddenLayer, vector<double>(sizeOutLayer)); //(������(�������))

		//������ - ���������� �������� ��������
		for (int i = 0; i < vWeightOutLayer.size(); i++)
		{
			//������� - ���������� �������� �������� ����
			for (int j = 0; j < vWeightOutLayer[i].size(); j++)
			{
				vWeightOutLayer[i][j] = 0.6;
			}
		}

		//for (int i = 0; i < vWeightOutLayer.size(); i++)
		//{
		//	for (int j = 0; j < vWeightOutLayer[i].size(); j++)
		//	{
		//		cout << vWeightOutLayer[i][j] << " ";
		//	}
		//	cout << endl;
		//}
		//cout << "���������� ����� ��������� ����: " << sizeWeightOutLayer << endl << endl;
	}

	//������� ���������� ������ ���� ����� � �������� ���� �� �����
	void createAllWeight()
	{
		vAllWeight.clear(); //������� ������ ����� �����������

		//�������� �� ���������� ������� ������� ����� 
		for (int i = 0; i < vWeightInLayer.size(); i++)
		{
			for (int j = 0; j < vWeightInLayer[i].size(); j++)
			{
				vAllWeight.push_back(vWeightInLayer[i][j]);
			}
		}
		//�������� �� ����������(�����������) ������� ����� �������� ���� 
		//for (int i = 0; i < minusFirstNumberHiddenLayer; i++)
		//{
			//������ - ���������� �������� � ����� ������� ����
		for (int j = 0; j < vWeightHiddenLayer.size(); j++)
		{
			//������� - ���������� �������� � ������ ������� ����
			for (int k = 0; k < vWeightHiddenLayer[j].size(); k++)
			{
				vAllWeight.push_back(vWeightHiddenLayer[j][k]);
			}
		}
		//}
		//�������� �� ���������� ������� ����� ��������� ����
		for (int i = 0; i < vWeightOutLayer.size(); i++)
		{
			for (int j = 0; j < vWeightOutLayer[i].size(); j++)
			{
				vAllWeight.push_back(vWeightOutLayer[i][j]);
			}
		}
		//for (int i = 0; i < vAllWeight.size(); i++)
		//{
		//	cout << vAllWeight[i] << " ";
		//}
		//cout << endl <<"��� ���� ������� ��������� � ����" << endl << endl;
	}

	//�������� ����������� ������� ���� ����� � ����
	void unloadWeightFile(string filename)
	{
		createAllWeight(); //������� ���������� ������ ���� ����� � �������� ���� �� �����
		unloadFile(filename, sizeAllWeight); //�������� ������ �� ����������� ������� � ����

		cout << endl << "�������� �������� ����� � ����!" << endl << endl;
	}

	//�������� ����� �� ����� � ���������� ������ 
	void loadWeightFile(string filename)
	{
		loadFile(filename, sizeAllWeight); //�������� ������ �� ����� � ���������� ������
		shareAllWeight(); //����� ���������� ������ ���� ����� �� ��������� �������

		cout << endl << "�������� �������� ����� �� �����!" << endl << endl;
	}

	//����� ���������� ������ ���� ����� �� ��������� �������
	void shareAllWeight()
	{
		int cVAW = 0; //������� ����������� ������� ���� �����

		//������ = ���������� ������� ����� ������ �� ���������� �������� �������� ����
		for (int i = 0; i < sizeInLayer; i++)
		{
			//������� - ���������� ������� ����� ������ �� ���������� ������� ��������
			for (int j = 0; j < sizeHiddenLayer; j++)
			{
				vWeightInLayer[i][j] = vAllWeight[cVAW];
				//cout << vAllWeight[cVAW] << " ";
				cVAW++;
			}
		/*	cout << endl;*/
		}
		//cout << endl;

		//������ - ������� ����� ����� ���� ������ ���� ������� �����
		//for (int i = 0; i < minusFirstNumberHiddenLayer; i++)
		//{
			//������ - ���������� �������� �������� ����
		for (int j = 0; j < minusFirstNumberHiddenLayer * sizeHiddenLayer; j++)
		{
			//������� - ���������� �������� ������� �������� ����
			for (int k = 0; k < sizeHiddenLayer; k++)
			{
				vWeightHiddenLayer[j][k] = vAllWeight[cVAW];
				cVAW++;
				cout << vWeightHiddenLayer[j][k] << " ";
			}
			cout << endl;
		}
		cout << endl;
		//}

		//������ - ���������� �������� ����� ������ �� ���������� �������� �������� ����
		for (int i = 0; i < sizeHiddenLayer; i++)
		{
			//������� - ���������� �������� ����� ������ �� ���������� �������� ��������
			for (int j = 0; j < sizeOutLayer; j++)
			{
				vWeightOutLayer[i][j] = vAllWeight[cVAW];
				cVAW++;
				cout << vWeightOutLayer[i][j] << " ";
			}
			cout << endl;
		}
		cout << endl;
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////

	//��������� �������� � �����, ��������� ���������� �������� ����� � ������, � ������� �������� � ����� �����
	void processingPrediction(int sizeLeft, int sizeRight, vector<double>vLeftLayer, vector<vector<double>>vLeftWeightLayer)
	{
		cout << endl << "������ ������������ �������� �� ����: " << endl;
		//1.������������ ����� �������� �� ����
		vMultiLayer.clear();
		int count = 0; //�������
		vMultiLayer.resize(sizeLeft*sizeRight); //������ ������ �������

		for (int i = 0; i < sizeLeft; i++) //������ ���������� �������� ������ ����
		{
			for (int j = 0; j < sizeRight; j++) //������� ���������� �������� ������� ����
			{
				vMultiLayer[count] = vLeftLayer[i] * vLeftWeightLayer[i][j];
				//cout << vMultiLayer[count] << " ";
				count++;
			}
		}
		//cout << endl << endl;

		cout << endl << "������ ���� ������������: " << endl;
		//2.����� ������������ ����� �������� �� ���� 
		vSumLayer.clear();
		vSumLayer.resize(sizeLeft*sizeRight); //������ ������ �������

		int counter = 0;
		//������ �� 0 �� ���������� �������� �������� ����
		for (int i = 0; i < sizeRight; i++)
		{
			//������� - �� 0 �� ���������� ������� ��������
			for (int j = 0; j < sizeLeft; j++)
			{
				vSumLayer[i] += vMultiLayer[counter];
			}
			//cout << vSumLayer[i] << " ";
		}
		//cout << endl << endl;

		cout << endl << "��������� ��������: " << endl;
		//3.��������� ���� � �����
		vActLayer.resize(sizeRight);

		for (int i = 0; i < vActLayer.size(); i++)
		{
			vActLayer[i] = sigmoid(vSumLayer[i]);
			//cout << vActLayer[i] << " ";
		}
		//cout << endl << endl;

		//4.���������� �������� ��������� � ������ ������
		for (int i = 0; i < vActLayer.size(); i++)
		{
			vAllActLayer.push_back(vActLayer[i]);
		}
	}

	//������������ ����
	void predictionNeuron()
	{
		cout << endl << "������������: " << endl << endl;
		vAllActLayer.clear(); //������� ������ ���� �������������� �������� ����� �������������
		vAllActLayer.shrink_to_fit(); //������� ������ ��������� �����

		cout << "1.������ �������� ������� �������� ����: " << endl;
		//1.�������� ��������� ��� �������� ����
		processingPrediction(sizeInLayer, sizeHiddenLayer, vInLayer, vWeightInLayer);

		cout << "2.������ �������� ��������� ������� ����: " << endl;
		//2.�������� ��������� ��� �������� ���� ������� ��� ������� ����� ����� ����
		if (minusFirstNumberHiddenLayer >= 1)
		{
			for (int i = 0; i < minusFirstNumberHiddenLayer; i++)
			{
				//��� ����� ����� ���������� ��������� ��������, � �������� ������� ����� �� �� ������������
				processingPrediction(sizeHiddenLayer, sizeHiddenLayer, vActLayer, vWeightHiddenLayer);
			}
		}

		cout << endl << "3.������ �������� ��������� ����: " << endl;
		//3.�������� ��������� ��� ��������� ����
		processingPrediction(sizeHiddenLayer, sizeOutLayer, vActLayer, vWeightOutLayer);

		//4.��������� �������������� ������� �������� ��������� �������� ���������
		for (int i = 0; i < sizeOutLayer; i++)
		{
			vOutLayer[i] = vActLayer[i];
			cout << vOutLayer[i] << endl;
		}

		////5.������� �������� ������� �����
		//cout << "�������� �������������� �������� ���� �����: " << endl;
		//for (int i = 0; i < vAllActLayer.size(); i++)
		//{
		//	cout << vAllActLayer[i] << " ";
		//}
		//cout << endl << endl;
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////

	//�������� ��������� ���� �������� � ����� ��� �������� �������� �������� � ���������� �� � ���������� �������
	void trainingNeuron()
	{
		cout << "��������: " << endl << endl;
		//�������� [�� ��������� ����� �� 0], 0 - ������� ��������, numberHiddenLayer - �������� ��������
		//����� ���������� ����� ������� ������ ������������ ��������
		s = numberHiddenLayer;
		while(s > -1) //�� ���������� ������� ����� �� 0
		{
			//cout << "�������� ��������: " << s << endl << endl;

			//���� �������� ��������
			if (s == numberHiddenLayer)
			{
				vTrainingLayer.resize(sizeWeightOutLayer);
				for (int i = 0; i < vOutLayer.size(); i++)
				{
					vTrainingLayer[i] = vOutLayer[i];
					//cout << vTrainingLayer[i] << " ";
				}
				//cout << endl << "������ ������ �������� ��������� ���� ����� � ��������!" << endl << endl;

				vTrainingWeight.resize(sizeOutLayer * sizeHiddenLayer);
				int count = 0; //�������
				for (int i = 0; i < vWeightOutLayer.size(); i++) //8 �����
				{
					for (int j = 0; j < vWeightOutLayer[i].size(); j++) //3 �������
					{
						vTrainingWeight[count] = vWeightOutLayer[i][j];
						//cout << vTrainingWeight[count] << " ";
						count++;
					}
					/*cout << endl;*/
				}
				//cout << "������ ����� ��������� ���� ����� � ��������!" << endl << endl;

				vTrainingActLayer.resize(sizeHiddenLayer);
				count = 0; //�������
				for (int i = sizeHiddenLayer * (s - 1); i < sizeHiddenLayer * s; i++)
				{
					vTrainingActLayer[count] = vAllActLayer[i];
					//cout << vTrainingActLayer[count] << " ";
					count++;
				}
				//cout << endl <<"������ ����� �������������� �������� ��������� ���� ����� � ��������!" << endl << endl;

				//��������: �������� �������� ������� � ����, � ����� �������� �������� �����
				trainingOutWeight(vTrainingLayer, vTrainingWeight, vTrainingActLayer);
			}

			//���� �������� ������� �����
			if (s > 0 && s < numberHiddenLayer)
			{
				iWeightHiddenMin = sizeHiddenLayer * (s - 1); //������ ������ ������� ����� ��� ������� ��������
				iWeightHiddenMax = sizeHiddenLayer * s; //��������� ������ ������� ����� ��� ������� ��������

				//cout << "������ ������ ���������: " << iWeightHiddenMin << endl;
				//cout << "��������� ������ ���������: " << iWeightHiddenMax << endl << endl;

				//������� ������� �� ���������� ������� �������� �������� ����
				vTrainingLayer.resize(sizeHiddenLayer); //���������� �������� ������ �������� ����
				int count = 0; //�������
				for (int i = s; i < s + 1; i++) //������������ ������ s 
				{
					for (int j = 0; j < sizeHiddenLayer; j++) //������� �� 0 �� 8
					{
						vTrainingLayer[count] = vHiddenLayer[i][j];
						//cout << vTrainingLayer[count] << " ";
						count++;
					}
					//cout << endl;
				}
				//cout << "������ ������ �������� �������� ���� ����� � ��������!" << endl << endl;

				//������� ���� �� ���������� ������� ����� �������� ���� 
				vTrainingWeight.resize(sizeHiddenLayer * sizeHiddenLayer); //���������� ����� ����� ����� �������� ������
				count = 0;
				for (int i = sizeHiddenLayer * (s - 1); i < sizeHiddenLayer * s; i++) //�� sizeHiddenLayer-1 �� sizeHiddenLayer
				{
					for (int j = 0; j < sizeHiddenLayer; j++) //������� ����� �� 0 �� 8
					{
						vTrainingWeight[count] = vWeightHiddenLayer[i][j];
						//cout << vTrainingWeight[count] << " ";
						count++;
					}
				/*	cout << endl;*/
				}
	/*			cout << "������ ����� �������� ���� ����� � ��������!" << endl << endl;*/

				//������� �������������� ������� �������� ���� 
				vTrainingActLayer.resize(sizeHiddenLayer);
				count = 0; //�������
				for (int i = sizeHiddenLayer * (s - 1); i < sizeHiddenLayer * s; i++) //�� sizeHiddenLayer-1 �� sizeHiddenLayer
				{
					vTrainingActLayer[count] = vAllActLayer[i];
		/*			cout << vTrainingActLayer[count] << " ";*/
					count++;
				}
		/*		cout << endl << "������ ����� �������������� �������� �������� ���� ����� � ��������!" << endl << endl;*/

				//��������: �������� ������� ������� � ����, � ����� �������� ������� �����
				trainingHiddenWeight(vTrainingLayer, vTrainingWeight, vTrainingActLayer);
			}

			//���� ������� ��������
			if (s == 0)
			{
				vTrainingLayer.resize(sizeInLayer);
				for (int i = 0; i < vInLayer.size(); i++)
				{
					vTrainingLayer[i] = vInLayer[i];
		/*			cout << vTrainingLayer[i] << " ";*/
				}
	/*			cout << endl << "������ ������ �������� �������� ���� ����� � ��������!" << endl << endl;*/

				vTrainingWeight.resize(sizeInLayer*sizeHiddenLayer);
				int count = 0;

				//������� ���� � ������� �������� ���� ��� ������������� ������� ��������
				for (int i = 0; i < vWeightInLayer.size(); i++)
				{
					for (int j = 0; j < vWeightInLayer[i].size(); j++)
					{
						vTrainingWeight[count] = vWeightInLayer[i][j];
			/*			cout << vTrainingWeight[count] << " ";*/
						count++;
					}
	/*				cout << endl;*/
				}
	/*			cout << "������ ����� �������� ���� ����� � ��������!" << endl << endl;*/

				//��������: �������� ������� ������� � ����, � ����� �������� ������� �����
				trainingInWeight(vTrainingLayer, vTrainingWeight, vTrainingActLayer);
			}
			s -= 1;
		}
	}

	//1.���������� ��������� �����
	void trainingOutWeight(vector<double> vTrainingLayer, vector<double> vTrainingWeight, vector<double> vTrainingActLayer)
	{
		//1.��� �������� �������� ������ ������ ����� �������� � ��������� ���������
		vErrLayer.resize(sizeOutLayer);

		for (int i = 0; i < vErrLayer.size(); i++) //������ ������� ������� � �������
		{
			vErrLayer[i] = vTrainingActLayer[i] - vConcretValues[i]; //������ �������������� �������� ����� ��������
			//cout << vErrLayer[i] << " ";
		}
		//cout << endl << "������ ������ ����� ��������� ���� ������ �������!" << endl;

		//3.������ ������ ��� �������� �����
		vDeltaWeight.resize(vErrLayer.size());

		for (int i = 0; i < vDeltaWeight.size(); i++) //����� ������� ������� � ������
		{
			vDeltaWeight[i] = vErrLayer[i] * sigmoidDx(vTrainingActLayer[i]); //�������� �� �������������� ��������
			//cout << vDeltaWeight[i] << " ";
		}
		//cout << endl << "������ ������� ����� ��������� ���� ������ �������!" << endl << endl;

		//4.������������� ���. �����(��� ������� ���� - �������� �������� ��������������� ������� * ������ ����� * ����������� ��������)
		for (int i = 0; i < sizeHiddenLayer; i++) //������ ���������� ������� ��������
		{
			for (int j = 0; j < sizeOutLayer; j++)  //������� ��������� ����� = ���������� �������� ��������
			{
				vWeightOutLayer[i][j] = vWeightOutLayer[i][j] - vTrainingActLayer[i] * vDeltaWeight[j] * learningRate;
				//cout << vWeightOutLayer[i][j] << " ";
			}
			//cout << endl;
		}
		//cout << "������������� �������� ����� ������ �������!" << endl << endl;
	}

	//2.���������� ����� �������� ����
	void trainingHiddenWeight(vector<double> vTrainingLayer, vector<double> vTrainingWeight, vector<double> vTrainingActLayer)
	{
		int counter = 0;

		//�������� �� ������ ������ �������� ����
		if (s == numberHiddenLayer - 1)
		{
			cout << "�������� ����� �������� ������� �����: " << endl << endl;
			//1.������������ ����� �������� ����� � ����� �������� �����
			vMultiErr.resize(sizeOutLayer*sizeHiddenLayer);

			//������������ ����� ����� ��������� ���� � ����� ��������� ����
			for (int i = 0; i < sizeHiddenLayer; i++) //���������� ������� ����� 
			{
				for (int j = 0; j < vDeltaWeight.size(); j++) //���������� �����
				{
					vMultiErr[counter] = vWeightOutLayer[i][j] * vDeltaWeight[j];
					//cout << vMultiErr[counter] << " ";
					counter++;
				}
				//cout << endl;
			}
			//cout << "������������ ����� �������� ����� � ����� �������� �����: " << vMultiErr.size() << endl << endl;
		}
		else
		{
			cout << "������� ����� �������� ������� �����: " << endl << endl;
			//1.������������ ����� �������� ����� � ����� �������� ����� 
			vMultiErr.resize(sizeHiddenLayer * sizeHiddenLayer);

			//������������ ����� ����� ��������� ���� � ����� ��������� ���� 
			for (int i = iWeightHiddenMin; i < iWeightHiddenMax; i++) //������ ���������� ������� ����� 
			{
				for (int j = 0; j < sizeHiddenLayer; j++) //������� ���������� ����� = ���������� �������� �������� ����
				{
					vMultiErr[counter] = vWeightHiddenLayer[i][j] * vDeltaWeight[j];
					//cout << vMultiErr[counter] << " ";
					counter++;
				}
		/*		cout << endl;*/
			}
			//cout << "������������ ����� ������� ����� � ����� ������� �����: " << vMultiErr.size() << endl << endl;
		}

		//2.������ �������� ���� = ������ ������������
		vSumErr.resize(sizeHiddenLayer); //������ ������� ����� ��� ����� ��������
		counter = 0;

		//������ �� 0 �� ���������� �������� �������� ����(���������� ������)
		for (int i = 0; i < sizeHiddenLayer; i++)
		{
			//������� - �� 0 �� ���������� ������� ��������(���������� ����)
			for (int j = 0; j < sizeHiddenLayer; j++)
			{
				vSumErr[i] += vMultiErr[counter];
			}
	/*		cout << vSumErr[i] << " ";*/
		}
		//cout << endl << "���������� ������ �������� ����: " << vSumErr.size() << endl << endl;

		//3.������ ������ ��� ����� �������� ����, ������ �������� �� ����������� ��������(�������� �������� ��������� ����)
		vDeltaWeight.resize(sizeHiddenLayer); //(���������� �������� � ����� ������� ����)

		for (int i = 0; i < vDeltaWeight.size(); i++) //(���������� �������� �������� ����)
		{
			vDeltaWeight[i] = vSumErr[i] * sigmoidDx(vTrainingActLayer[i]); //�������� �� �������������� ��������
			//cout << vDeltaWeight[i] << " ";
		}
		//cout << endl << "���������� ����� �������� ����: " << vDeltaWeight.size() << endl << endl;

		//4.������������� �����(��� �������� ���� - �������� �������� ��������������� ������� * ������ ����� * ����������� ��������)
		vTrainingBufWeight.resize(sizeHiddenLayer*sizeHiddenLayer);
		counter = 0;
		for (int i = 0; i < sizeHiddenLayer; i++) //���������� �����
		{
			for (int j = 0; j < sizeHiddenLayer; j++) //���������� �������������� ��������
			{
				//����� ���� ���������� � ����� �����
				vTrainingBufWeight[counter] = vTrainingWeight[counter] - vTrainingActLayer[j] * vDeltaWeight[i] * learningRate;
				counter++;
			}
		}
		//�� ������ �����, ��������� � ��������� ������ ����� � ���������� �������
		counter = 0;
		for (int i = iWeightHiddenMin; i < iWeightHiddenMax; i++) //�� ������ �� ��������� ������ �������� �������� ����
		{
			for (int j = 0; j < sizeHiddenLayer; j++) //���������� �������� �������� ����
			{
				vWeightHiddenLayer[i][j] = vTrainingBufWeight[counter];
				//cout << vWeightHiddenLayer[i][j] << " ";
				counter++;
			}
			cout << endl;
		}
		//cout << "������������� ����� �������� ���� ������ �������!" << endl << endl;
	}

	//3.���������� ����� �������� ����
	void trainingInWeight(vector<double> vTrainingLayer, vector<double> vTrainingWeight, vector<double> vTrainingActLayer)
	{
		//1.������������ ����� ����� � ����� �������� ����
		vMultiErr.resize(sizeInLayer*sizeHiddenLayer);
		int counter = 0;
		for (int i = 0; i < sizeInLayer; i++) //���������� �������� �������� ���� = ���������� ���� � ������
		{
			for (int j = 0; j < vDeltaWeight.size(); j++) //���������� ����� = ���������� �������� � ������� ����
			{
				vMultiErr[counter] = vWeightInLayer[i][j] * vDeltaWeight[j];
				//cout << vMultiErr[counter] << " ";
				counter++;
			}
			//cout << endl;
		}
		//cout << "������������ ����� ������� ����� � ����� ������� �����: " << vMultiErr.size() << endl << endl;

		//2.������ �������� ���� = ������ ������������
		vSumErr.resize(sizeHiddenLayer); //������ ������� ����� ��� ����� ��������
		counter = 0;

		//������ �� 0 �� ���������� �������� �������� ����(���������� ������)
		for (int i = 0; i < sizeHiddenLayer; i++)
		{
			//������� - �� 0 �� ���������� ������� ��������(���������� ����)
			for (int j = 0; j < sizeHiddenLayer; j++)
			{
				vSumErr[i] += vMultiErr[counter];
			}
			//cout << vSumErr[i] << " ";
		}
		//cout << endl << "���������� ������ �������� ����: " << vSumErr.size() << endl << endl;

		//3.������ ������ ��� ����� �������� ����, ������ �������� �� ����������� ��������(�������� �������� ��������� ����)
		vDeltaWeight.resize(sizeHiddenLayer); //(���������� �������� � ����� ������� ����)

		for (int i = 0; i < vDeltaWeight.size(); i++) //(���������� �������� �������� ����)
		{
			vDeltaWeight[i] = vSumErr[i] * sigmoidDx(vTrainingActLayer[i]); //�������� ��������������� ������� 1 �������� ����
			//cout << vDeltaWeight[i] << " ";
		}
		//cout << endl << "���������� ����� �������� ����: " << vDeltaWeight.size() << endl << endl;

		//4.������������� �����(��� �������� ���� - �������� �������� ������� * ������ ����� * ����������� ��������)
		vTrainingBufWeight.resize(sizeInLayer*sizeHiddenLayer);
		counter = 0;
		for (int i = 0; i < sizeHiddenLayer; i++) //���������� �����
		{
			for (int j = 0; j < sizeInLayer; j++) //���������� ������� ��������
			{
				vTrainingBufWeight[counter] = vTrainingWeight[counter] - vInLayer[j] * vDeltaWeight[i] * learningRate;
				counter++;
			}
		}
		//5.�� ������ �����, ��������� � ��������� ������ ����� � ���������� �������
		counter = 0;
		for (int i = 0; i < sizeInLayer; i++) //�� ������ �� ��������� ������ �������� �������� ����
		{
			for (int j = 0; j < sizeHiddenLayer; j++) //���������� �������� �������� ����
			{
				vWeightInLayer[i][j] = vTrainingBufWeight[counter];
				//cout << vWeightInLayer[i][j] << " ";
				counter++;
			}
			//cout << endl;
		}
		//cout << "������������� ����� �������� ���� ������ �������!" << endl << endl;
	}

};
