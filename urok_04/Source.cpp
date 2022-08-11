#include "Neuron.h"

int sIn = 2; //входные нейроны
int nHidden = 1; //скрытые слои
int sHidden = 8; //нейроны скрытого слоя
int sOut = 3; //выходные нейроны
int epochs = 1000; //количество эпох обучения

double Ht = 1000; //текущая высота
double Hz = 2000; //заданная высота

vector<double> InLayer; //вектор входных нейронов
vector<double> Concret; //вектор выходных эталонных значений

//ф-я эталонных значений, автоматически выбирает правильный ответ
void fСreateConcret(Neuron n)
{
	int ansver = 0;

	if (Ht < 1800) ansver = 1;
	else if (Ht > 1799 && Ht < 2201) ansver = 2;
	else if (Ht > 2200) ansver = 3;
	cout << endl << "Правильный режим: " << ansver << endl;

	Concret.clear(); //очистка
	Concret.shrink_to_fit(); //очистка резерва

	switch (ansver)
	{
	case 1: //если ответ первый
		//вектор заполняется 1 0 0
		Concret.push_back(1);
		Concret.push_back(0);
		Concret.push_back(0);

	case 2: //если ответ второй 0 1 0
		Concret.push_back(0);
		Concret.push_back(1);
		Concret.push_back(0);


	case 3: //если ответ третий 0 0 1
		Concret.push_back(0);
		Concret.push_back(0);
		Concret.push_back(1);

	}

	//1.создает вектор эталонных выходных значений
	n.createConcretValues(Concret);
}

//1)создание нейросети
void fcreateNeuron(Neuron n)
{
	//-очистка вектор и резервная очистка
	InLayer.clear(); //очистка вектора
	InLayer.shrink_to_fit(); //очистка резервных мест

	//-пушбечит вектор заданными входными параметрами
	InLayer.push_back(Hz / 3000);
	InLayer.push_back(Ht / 3000);

	//1.создает одномерный вектор входного слоя
	n.createInLayer(InLayer); 

	//2.создает двумерный вектор нейронов скрытых слоев
	n.createHiddenLayer();
	
	//3.создает одномерный вектор нейронов выходного слоя
	n.createOutLayer(); //создает вектор выходного слоя
	
	//4.создает двумерный вектор весов входного слоя
	n.createWeightInLayer(); //создает вектор весов входного слоя

	//5.создает двумерный вектор весов скрытых слоев
	n.createWeightHiddenLayer(); //создает вектор весов скрытого слоя

	//6.создает двумерный вектор весов выходного слоя
	n.createWeightOutLayer(); //создает вектор весов выходного слоя

	//7.все веса записывает в одномерный вектор и выгружает в файл
	n.unloadWeightFile("weight.txt"); //выгружает одномерный вектор всех весов в файл
}

//2)предсказания нейросети
void fPredictionNeuron(Neuron n)
{
	cout << endl << "НАЧАЛО ПРЕДСКАЗАНИЯ!" << endl << endl;
	//1.загружает одномерный вектор весов из файла и распределяет по двумерным векторам весов
	n.loadWeightFile("weight.txt"); //загружает весов в один вектор

	//2.ф-я задает эталонные значения
	fСreateConcret(n);

	//3.используя входные значения и веса, делает предсказание и выводит выходные значения в столбик
	n.predictionNeuron(); //ф-я предсказания
}

//3)обучение нейросети
void fTrainingNeuron(Neuron n, int epochs)
{
	cout << endl << "НАЧАЛО ОБУЧЕНИЯ!" << endl << endl;

	//1.загружает одномерный вектор весов из файла и распределяет по двумерным векторам весов
	n.loadWeightFile("weight.txt"); //загрузка все веса в один вектор

	for (int i = 1; i < epochs; i++)
	{
		cout << "Эпоха Обучения: " << i << " ===========================================" <<endl << endl;
		//1.генерирует разные входные значения
		Ht = rand() % 2000 + 1000; //изменяем текущую высоты

		//2.заполняет одномерный вектор нейронов входного слоя
		InLayer.clear(); //очистка
		InLayer.shrink_to_fit(); //очистка резервных мест

		InLayer.push_back(Hz/3000);
		InLayer.push_back(Ht/3000);

		//4.выводит пользователю входные значения 
		cout << "Заданная высота = " << Hz << endl;
		cout << "Текущая высота = " << Ht << endl << endl;

		//4.передает новые параметры входного слоя
		n.createInLayer(InLayer);

		//5.ф-я задает эталонные значения
		fСreateConcret(n);

		//6.используя входные значения и веса, делает предсказание и выводит выходные значения в столбик
		n.predictionNeuron(); //ф-я предсказания

		//7.делает обучение, по текущим весам и активированным нейронам предсказания
		n.trainingNeuron(); //ф-я обучения
	}

	//6.все веса записывает в одномерный вектор и выгружает в файл
	n.unloadWeightFile("weight.txt"); //выгружает вектор всех весов в файл
}


int main()
{
	setlocale(LC_ALL, "ru");
	srand(NULL);

	Neuron n(sIn, nHidden, sHidden, sOut); //передает параметры нейросети

	fcreateNeuron(n); //создает нейросет
	system("pause");

	fTrainingNeuron(n, epochs); //обучение нейросети
	system("pause");

	int end = 1;
	while (end != 0)
	{
		Ht = rand() % 2000 + 1000; //изменяем текущую высоты

		cout << endl << "Заданная высота = " << Hz << endl;
		cout << "Текущая высота = " << Ht << endl << endl;

		InLayer.clear(); //очистка
		InLayer.shrink_to_fit(); //очистка резервных мест

		InLayer.push_back(Hz/3000);
		InLayer.push_back(Ht/3000);

		//2.передает новые параметры входного слоя
		n.createInLayer(InLayer);

		//3.предсказание нейросети
		fPredictionNeuron(n); 

		//cout << "Для завершения программы введите 0: ";
		//cin >> end;
		cout << endl << endl;
		system("pause");
	};


	return 0;
}

