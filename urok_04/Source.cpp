#include "Neuron.h"

int sIn = 2; //������� �������
int nHidden = 1; //������� ����
int sHidden = 8; //������� �������� ����
int sOut = 3; //�������� �������
int epochs = 1000; //���������� ���� ��������

double Ht = 1000; //������� ������
double Hz = 2000; //�������� ������

vector<double> InLayer; //������ ������� ��������
vector<double> Concret; //������ �������� ��������� ��������

//�-� ��������� ��������, ������������� �������� ���������� �����
void f�reateConcret(Neuron n)
{
	int ansver = 0;

	if (Ht < 1800) ansver = 1;
	else if (Ht > 1799 && Ht < 2201) ansver = 2;
	else if (Ht > 2200) ansver = 3;
	cout << endl << "���������� �����: " << ansver << endl;

	Concret.clear(); //�������
	Concret.shrink_to_fit(); //������� �������

	switch (ansver)
	{
	case 1: //���� ����� ������
		//������ ����������� 1 0 0
		Concret.push_back(1);
		Concret.push_back(0);
		Concret.push_back(0);

	case 2: //���� ����� ������ 0 1 0
		Concret.push_back(0);
		Concret.push_back(1);
		Concret.push_back(0);


	case 3: //���� ����� ������ 0 0 1
		Concret.push_back(0);
		Concret.push_back(0);
		Concret.push_back(1);

	}

	//1.������� ������ ��������� �������� ��������
	n.createConcretValues(Concret);
}

//1)�������� ���������
void fcreateNeuron(Neuron n)
{
	//-������� ������ � ��������� �������
	InLayer.clear(); //������� �������
	InLayer.shrink_to_fit(); //������� ��������� ����

	//-�������� ������ ��������� �������� �����������
	InLayer.push_back(Hz / 3000);
	InLayer.push_back(Ht / 3000);

	//1.������� ���������� ������ �������� ����
	n.createInLayer(InLayer); 

	//2.������� ��������� ������ �������� ������� �����
	n.createHiddenLayer();
	
	//3.������� ���������� ������ �������� ��������� ����
	n.createOutLayer(); //������� ������ ��������� ����
	
	//4.������� ��������� ������ ����� �������� ����
	n.createWeightInLayer(); //������� ������ ����� �������� ����

	//5.������� ��������� ������ ����� ������� �����
	n.createWeightHiddenLayer(); //������� ������ ����� �������� ����

	//6.������� ��������� ������ ����� ��������� ����
	n.createWeightOutLayer(); //������� ������ ����� ��������� ����

	//7.��� ���� ���������� � ���������� ������ � ��������� � ����
	n.unloadWeightFile("weight.txt"); //��������� ���������� ������ ���� ����� � ����
}

//2)������������ ���������
void fPredictionNeuron(Neuron n)
{
	cout << endl << "������ ������������!" << endl << endl;
	//1.��������� ���������� ������ ����� �� ����� � ������������ �� ��������� �������� �����
	n.loadWeightFile("weight.txt"); //��������� ����� � ���� ������

	//2.�-� ������ ��������� ��������
	f�reateConcret(n);

	//3.��������� ������� �������� � ����, ������ ������������ � ������� �������� �������� � �������
	n.predictionNeuron(); //�-� ������������
}

//3)�������� ���������
void fTrainingNeuron(Neuron n, int epochs)
{
	cout << endl << "������ ��������!" << endl << endl;

	//1.��������� ���������� ������ ����� �� ����� � ������������ �� ��������� �������� �����
	n.loadWeightFile("weight.txt"); //�������� ��� ���� � ���� ������

	for (int i = 1; i < epochs; i++)
	{
		cout << "����� ��������: " << i << " ===========================================" <<endl << endl;
		//1.���������� ������ ������� ��������
		Ht = rand() % 2000 + 1000; //�������� ������� ������

		//2.��������� ���������� ������ �������� �������� ����
		InLayer.clear(); //�������
		InLayer.shrink_to_fit(); //������� ��������� ����

		InLayer.push_back(Hz/3000);
		InLayer.push_back(Ht/3000);

		//4.������� ������������ ������� �������� 
		cout << "�������� ������ = " << Hz << endl;
		cout << "������� ������ = " << Ht << endl << endl;

		//4.�������� ����� ��������� �������� ����
		n.createInLayer(InLayer);

		//5.�-� ������ ��������� ��������
		f�reateConcret(n);

		//6.��������� ������� �������� � ����, ������ ������������ � ������� �������� �������� � �������
		n.predictionNeuron(); //�-� ������������

		//7.������ ��������, �� ������� ����� � �������������� �������� ������������
		n.trainingNeuron(); //�-� ��������
	}

	//6.��� ���� ���������� � ���������� ������ � ��������� � ����
	n.unloadWeightFile("weight.txt"); //��������� ������ ���� ����� � ����
}


int main()
{
	setlocale(LC_ALL, "ru");
	srand(NULL);

	Neuron n(sIn, nHidden, sHidden, sOut); //�������� ��������� ���������

	fcreateNeuron(n); //������� ��������
	system("pause");

	fTrainingNeuron(n, epochs); //�������� ���������
	system("pause");

	int end = 1;
	while (end != 0)
	{
		Ht = rand() % 2000 + 1000; //�������� ������� ������

		cout << endl << "�������� ������ = " << Hz << endl;
		cout << "������� ������ = " << Ht << endl << endl;

		InLayer.clear(); //�������
		InLayer.shrink_to_fit(); //������� ��������� ����

		InLayer.push_back(Hz/3000);
		InLayer.push_back(Ht/3000);

		//2.�������� ����� ��������� �������� ����
		n.createInLayer(InLayer);

		//3.������������ ���������
		fPredictionNeuron(n); 

		//cout << "��� ���������� ��������� ������� 0: ";
		//cin >> end;
		cout << endl << endl;
		system("pause");
	};


	return 0;
}

