#pragma once

#include "lib.h"

vector<double> vAllWeight; //���������� ������ ���� �����

//�������� ������ �� ����� � ���������� ������
void loadFile(string filename, int sizeAllWeight)
{
	vAllWeight.clear(); //������� ������
	vAllWeight.shrink_to_fit(); //������� ������ ��������� �����

	double vectorbuf; //����� ��������� �������

	//��������� ���� � �����
	ifstream ifile(filename);

	if (ifile.is_open())
	{
		cout << "���� ������. " << endl;
		while (ifile >> vectorbuf)
		{
			//���������� ������ � ������
			vAllWeight.push_back(vectorbuf);
		}
	}
	else
	{
		cout << "�� ������� ������� ����." << endl;
	}
	ifile.close();

	//cout << "\n\n���������� �����: " << vAllWeight.size() << "\n\n";
}

//�������� ������ �� ����������� ������� � ���� //vbuf - ����� ������ ����� ���������, sizeAllWeight - ���������� ���� �����
void unloadFile(string filename, int sizeAllWeight) 
{
	//������� txt ����
	ofstream ofile(filename);

	vAllWeight.resize(sizeAllWeight); //����������� ����� � ������� ��� ���� �����

	//��������� ������������ ��� ������ � ������
	for (int i = 0; i < vAllWeight.size(); i++)
	{
		//for (int j = 0; j < 1; j++)
		//{
			//���������� ������� ������� ����� � ����
			ofile << vAllWeight[i] << " ";
		//}
		//��������� ������� � ����� �������
		//vWeight.push_back(vbuf);
		//��������� ������ ������� ������ ���������
		//vWeight[i] = vbuf;
	}
	ofile.close();
}

