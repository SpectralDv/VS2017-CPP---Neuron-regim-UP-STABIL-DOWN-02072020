#pragma once

#include "txtReader.h"

vector<double> vInLayer; //вектор входных нейронов
vector<double> vOutLayer; //вектор выходных нейронов
vector<vector<double>> vHiddenLayer; //двумерный вектор нейронов скрытого слоя

vector<vector<double>> vWeightInLayer; //двумерный вектор весов входного слоя
vector<vector<double>> vWeightHiddenLayer; //трехмерный вектор весов скрытых слоев
vector<vector<double>> vWeightOutLayer; //двумерный вектор весов выходного слоя

vector<double> vMultiLayer; //произведение входных нейронов на входные веса
vector<double> vSumLayer; //вектор суммы произведений
vector<double> vActLayer; //вектор активации 
vector<double> vAllActLayer; //вектор всех активаций

vector<double> vConcretValues; //вектор эталонных значений
vector<double> vErrLayer; //вектор ошибок
vector<double> vDeltaWeight; //вектор сдвигов весов
double learningRate = 0.5; //коэффициент обучения
vector<double> vMultiErr; //вектор произведений новых весов и дельт весов
vector<double> vSumErr; //вектор сумм произведений

vector<double> vTrainingLayer; //вектор текущий нейронов для обучения
vector<double> vTrainingWeight; //вектор текущих весов для обучения
vector<double> vTrainingActLayer; //вектор активированных нейронов для обучения
vector<double> vTrainingBufWeight; //вектор буфера нейронов для обучения

class Neuron
{
	int sizeInLayer; //количество входных нейронов
	int numberHiddenLayer; //количество скрытых слоев
	int minusFirstNumberHiddenLayer;//количество скрытых слоев для весов скрытого слоя
	int sizeHiddenLayer; //количество нейронов одного скрытого слоя
	int sizeOutLayer; //количество выходных нейронов
	int sizeWeightInLayer; //количество входных весов
	int sizeWeightHiddenLayer; //количество весов скрытого слоя
	int sizeWeightOutLayer; //количество выходных весов
	int sizeAllWeight; //количество всех весов
	int iterationTraining; //хранит текущую итерацию обучения
	int iWeightHiddenMin; //итератор по двумерному вектору скрытого слоя
	int iWeightHiddenMax; //ограничение по максимальному значению для iWeightHidden
	int s; //счетчик итератор обучения

	double correctAnsver = 0; //счетчик правильных ответов для показателя качества
	double epochMSE = 1; //счетчик эпох от 1 до 100, 101 = 1
	int counterMSE = 0; //процент ошибок при обучении

public:

	//конструктор принимает количество входных нейронов, скрытых слоев, нейронов скрытых слоев, выходных нейронов
	Neuron(int sInLayer, int nHiddenLayer, int sHiddenLayer, int sOutLayer)
	{
		sizeInLayer = sInLayer;
		numberHiddenLayer = nHiddenLayer;
		minusFirstNumberHiddenLayer = nHiddenLayer - 1;
		sizeHiddenLayer = sHiddenLayer;
		sizeOutLayer = sOutLayer;

		//нумерация весов входного слоя от 0 до sInLayer
		sizeWeightInLayer = sizeInLayer * sizeHiddenLayer;
		//нумерация весов скрытого слоя от sInLayer до sHiddenLayer
		sizeWeightHiddenLayer = minusFirstNumberHiddenLayer * (sizeHiddenLayer * sizeHiddenLayer);
		//нумерация весов выходного слоя от sHiddenLayer до sOutLayer
		sizeWeightOutLayer = sizeHiddenLayer * sizeOutLayer;

		sizeAllWeight = sizeInLayer * sizeHiddenLayer + minusFirstNumberHiddenLayer * (sizeHiddenLayer * sizeHiddenLayer) + sizeHiddenLayer * sizeOutLayer;

		s = numberHiddenLayer; //итератор обучения слоев = количеству скрытых слев

		cout << "Создание параметров нейросети прошло успешно!" << endl << endl;
	}

	//ф-я сигмоиды
	double sigmoid(double x)
	{
		double s;
		s = 1 / (1 + exp(-x));
		return s;
	}
	//ф-я производная сигмоды
	double sigmoidDx(double x)
	{
		double sDx;
		sDx = sigmoid(x) * (1 - sigmoid(x));
		return sDx;
	}

	//задает эталонные занчения
	void createConcretValues(vector<double> a)
	{
		vConcretValues.resize(sizeOutLayer);

		cout << "Эталонные значения: ";
		for (int i = 0; i < vConcretValues.size(); i++)
		{
			vConcretValues[i] = a[i];
			cout << vConcretValues[i] << " ";
		}
		//cout << endl <<"Создание выходных эталонных значений прошло успешно!" << endl << endl;
	}

	//создает вектор входных нейронов
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
		cout << endl << "Создание одномерного вектора входных нейронов прошло успешно!" << endl << endl;*/
	}

	//создает вектор выходных нейронов
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
		//cout << endl << "Создание одномерного вектора выходных нейронов прошло успешно!" << endl << endl;
	}

	//создаем двумерный вектор скрытых слоев и нейронов в них
	void createHiddenLayer()
	{
		vHiddenLayer.assign(numberHiddenLayer, vector<double>(sizeHiddenLayer)); //(строка(столбец))

		//строка - количество скрытых слоев
		for (int i = 0; i < vHiddenLayer.size(); i++)
		{
			//столбец - количество нейронов в одном скрытом слое
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
		//cout << endl << "Создание одномерного вектора скрытых слоев прошло успешно!" << endl << endl;
	}

	//создает двумерный ветор весов входного слоя
	void createWeightInLayer()
	{
		vWeightInLayer.assign(sizeInLayer, vector<double>(sizeHiddenLayer, 0.5)); //(строка(столбец))

		//строка - количество входных нейронов
		for (int i = 0; i < vWeightInLayer.size(); i++)
		{
			//столбец - количество скрытых слоев
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

		//cout << "Количество весов входного слоя: " << sizeWeightInLayer << endl << endl;
	}

	//создает трехмерный вектор скрытых нейронов и заполняет одинаковыми значениями
	void createWeightHiddenLayer()
	{
		//vWeightHiddenLayer.assign(minusFirstNumberHiddenLayer, vector<vector<double>>(sizeHiddenLayer, vector<double>(sizeHiddenLayer)));

		vWeightHiddenLayer.assign(sizeHiddenLayer*minusFirstNumberHiddenLayer, vector<double>(sizeHiddenLayer));

		//индекс - количество векторов весов скрытого слоя в векторе скрытых слоев
		//for (int i = 0; i < minusFirstNumberHiddenLayer; i++)
		//{
			//строка - количество нейронов в одном скрытом слое
		for (int j = 0; j < sizeHiddenLayer*minusFirstNumberHiddenLayer; j++)
		{
			//столбец - количество нейронов в другом скрытом слое
			for (int k = 0; k < sizeHiddenLayer; k++)
			{
				vWeightHiddenLayer[j][k] = 0.5;
			}
		}
		//}

	//	//for (int i = 0; i < minusFirstNumberHiddenLayer; i++)
	//	//{
	//		//строка - количество нейронов в одном скрытом слое
	//	for (int j = 0; j < vWeightHiddenLayer.size(); j++)
	//	{
	//		//столбец - количество нейронов в другом скрытом слое
	//		for (int k = 0; k < vWeightHiddenLayer[j].size(); k++)
	//		{
	//			cout << vWeightHiddenLayer[j][k] << " ";
	//		}
	//		cout << endl;
	//	}
	//	//cout << endl;
	////}
	//	cout << "Количество весов скрытого слоя: " << sizeWeightHiddenLayer << endl << endl;
	}

	//создает двумерный вектор весов выходных нейронов
	void createWeightOutLayer()
	{
		vWeightOutLayer.assign(sizeHiddenLayer, vector<double>(sizeOutLayer)); //(строка(столбец))

		//строка - количество выходных нейронов
		for (int i = 0; i < vWeightOutLayer.size(); i++)
		{
			//столбец - количество нейронов скрытого слоя
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
		//cout << "Количество весов выходного слоя: " << sizeWeightOutLayer << endl << endl;
	}

	//создает одномерный вектор всех весов и забирает веса со слоев
	void createAllWeight()
	{
		vAllWeight.clear(); //очищает вектор перед заполнением

		//проходит по двумерному вектору входных весов 
		for (int i = 0; i < vWeightInLayer.size(); i++)
		{
			for (int j = 0; j < vWeightInLayer[i].size(); j++)
			{
				vAllWeight.push_back(vWeightInLayer[i][j]);
			}
		}
		//проходит по двумерному(трехмерному) вектору весов скрытого слоя 
		//for (int i = 0; i < minusFirstNumberHiddenLayer; i++)
		//{
			//строка - количество нейронов в одном скрытом слое
		for (int j = 0; j < vWeightHiddenLayer.size(); j++)
		{
			//столбец - количество нейронов в другом скрытом слое
			for (int k = 0; k < vWeightHiddenLayer[j].size(); k++)
			{
				vAllWeight.push_back(vWeightHiddenLayer[j][k]);
			}
		}
		//}
		//проходит по двумерному вектору весов выходного слоя
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
		//cout << endl <<"Все веса которые загружены в файл" << endl << endl;
	}

	//выгрузка одномерного вектора всех весов в файл
	void unloadWeightFile(string filename)
	{
		createAllWeight(); //создает одномерный вектор всех весов и забирает веса со слоев
		unloadFile(filename, sizeAllWeight); //выгрузка данных из одномерного вектора в файл

		cout << endl << "Успешная выгрузка весов в файл!" << endl << endl;
	}

	//загрузка весов из файла в одномерный вектор 
	void loadWeightFile(string filename)
	{
		loadFile(filename, sizeAllWeight); //загрузка данных из файла в одномерный вектор
		shareAllWeight(); //делит одномерный вектор всех весов на отдельные вектора

		cout << endl << "Успешная загрузка весов из файла!" << endl << endl;
	}

	//делит одномерный вектор всех весов на отдельные вектора
	void shareAllWeight()
	{
		int cVAW = 0; //счетчик одномерного вектора всех весов

		//строка = количество входных весов делить на количество нейронов скрытого слоя
		for (int i = 0; i < sizeInLayer; i++)
		{
			//столбец - количество входных весов делить на количество входных нейронов
			for (int j = 0; j < sizeHiddenLayer; j++)
			{
				vWeightInLayer[i][j] = vAllWeight[cVAW];
				//cout << vAllWeight[cVAW] << " ";
				cVAW++;
			}
		/*	cout << endl;*/
		}
		//cout << endl;

		//индекс - сколько групп весов если больше двух скрытых слоев
		//for (int i = 0; i < minusFirstNumberHiddenLayer; i++)
		//{
			//строка - количество нейронов скрытого слоя
		for (int j = 0; j < minusFirstNumberHiddenLayer * sizeHiddenLayer; j++)
		{
			//столбец - количество нейронов другого скрытого слоя
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

		//строка - количество выходных весов делить на количество нейронов скрытого слоя
		for (int i = 0; i < sizeHiddenLayer; i++)
		{
			//столбец - количество выходных весов делить на количество выходных нейронов
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

	//обработка нейронов и весов, принимает количество нейронов слева и справа, и вектора нейронов и весов слева
	void processingPrediction(int sizeLeft, int sizeRight, vector<double>vLeftLayer, vector<vector<double>>vLeftWeightLayer)
	{
		cout << endl << "Расчет произведений нейронов на выса: " << endl;
		//1.произведение левых нейронов на веса
		vMultiLayer.clear();
		int count = 0; //счетчик
		vMultiLayer.resize(sizeLeft*sizeRight); //задает размер вектора

		for (int i = 0; i < sizeLeft; i++) //строки количество нейронов левого слоя
		{
			for (int j = 0; j < sizeRight; j++) //столбцы количество нейронов правого слоя
			{
				vMultiLayer[count] = vLeftLayer[i] * vLeftWeightLayer[i][j];
				//cout << vMultiLayer[count] << " ";
				count++;
			}
		}
		//cout << endl << endl;

		cout << endl << "Расчет сумм произведений: " << endl;
		//2.сумма произведений левых нейронов на веса 
		vSumLayer.clear();
		vSumLayer.resize(sizeLeft*sizeRight); //задает размер вектора

		int counter = 0;
		//строки от 0 до количества нейронов скрытого слоя
		for (int i = 0; i < sizeRight; i++)
		{
			//столбцы - от 0 до количество входных нейронов
			for (int j = 0; j < sizeLeft; j++)
			{
				vSumLayer[i] += vMultiLayer[counter];
			}
			//cout << vSumLayer[i] << " ";
		}
		//cout << endl << endl;

		cout << endl << "Активация нейронов: " << endl;
		//3.активация сумм и вывод
		vActLayer.resize(sizeRight);

		for (int i = 0; i < vActLayer.size(); i++)
		{
			vActLayer[i] = sigmoid(vSumLayer[i]);
			//cout << vActLayer[i] << " ";
		}
		//cout << endl << endl;

		//4.запоминает значение активации в другой вектор
		for (int i = 0; i < vActLayer.size(); i++)
		{
			vAllActLayer.push_back(vActLayer[i]);
		}
	}

	//предсказание сети
	void predictionNeuron()
	{
		cout << endl << "Предсказание: " << endl << endl;
		vAllActLayer.clear(); //очистка буфера всех активированных нейронов перед предсказанием
		vAllActLayer.shrink_to_fit(); //удаляет пустые резервные места

		cout << "1.Расчет нейронов первого скрытого слоя: " << endl;
		//1.вызывает обработку для входного слоя
		processingPrediction(sizeInLayer, sizeHiddenLayer, vInLayer, vWeightInLayer);

		cout << "2.Расчет нейронов следующих скрытых слов: " << endl;
		//2.вызывает обработку для скрытого слоя столько раз сколько слоев минус один
		if (minusFirstNumberHiddenLayer >= 1)
		{
			for (int i = 0; i < minusFirstNumberHiddenLayer; i++)
			{
				//все равно какой передавать последний параметр, в расчетах скрытых слоев он не используется
				processingPrediction(sizeHiddenLayer, sizeHiddenLayer, vActLayer, vWeightHiddenLayer);
			}
		}

		cout << endl << "3.Расчет нейронов выходного слоя: " << endl;
		//3.вызывает обработку для выходного слоя
		processingPrediction(sizeHiddenLayer, sizeOutLayer, vActLayer, vWeightOutLayer);

		//4.последние активированные нейорны являются выходными занчения нейросети
		for (int i = 0; i < sizeOutLayer; i++)
		{
			vOutLayer[i] = vActLayer[i];
			cout << vOutLayer[i] << endl;
		}

		////5.выводит активные нейроны слоев
		//cout << "Значения активированных нейронов всех слоев: " << endl;
		//for (int i = 0; i < vAllActLayer.size(); i++)
		//{
		//	cout << vAllActLayer[i] << " ";
		//}
		//cout << endl << endl;
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////

	//выбирает следующий слой нейронов и весов для текущеой итерации обучения и записывает их в одномерные векторы
	void trainingNeuron()
	{
		cout << "Обучение: " << endl << endl;
		//итерации [от количетва слоев до 0], 0 - входная итерация, numberHiddenLayer - выходная итерация
		//нужно определить какую следует сейчас обрабатывать итерацию
		s = numberHiddenLayer;
		while(s > -1) //от количества скрытых слоев до 0
		{
			//cout << "Итерация обучения: " << s << endl << endl;

			//если выходная итерация
			if (s == numberHiddenLayer)
			{
				vTrainingLayer.resize(sizeWeightOutLayer);
				for (int i = 0; i < vOutLayer.size(); i++)
				{
					vTrainingLayer[i] = vOutLayer[i];
					//cout << vTrainingLayer[i] << " ";
				}
				//cout << endl << "Вектор правых нейронов выходного слоя готов к обучению!" << endl << endl;

				vTrainingWeight.resize(sizeOutLayer * sizeHiddenLayer);
				int count = 0; //счетчик
				for (int i = 0; i < vWeightOutLayer.size(); i++) //8 строк
				{
					for (int j = 0; j < vWeightOutLayer[i].size(); j++) //3 столбца
					{
						vTrainingWeight[count] = vWeightOutLayer[i][j];
						//cout << vTrainingWeight[count] << " ";
						count++;
					}
					/*cout << endl;*/
				}
				//cout << "Вектор весов выходного слоя готов к обучению!" << endl << endl;

				vTrainingActLayer.resize(sizeHiddenLayer);
				count = 0; //счетчик
				for (int i = sizeHiddenLayer * (s - 1); i < sizeHiddenLayer * s; i++)
				{
					vTrainingActLayer[count] = vAllActLayer[i];
					//cout << vTrainingActLayer[count] << " ";
					count++;
				}
				//cout << endl <<"Вектор левых активированных нейронов выходного слоя готов к обучению!" << endl << endl;

				//ОБУЧЕНИЕ: передаем выходные нейроны и веса, и буфер хранения выходных весов
				trainingOutWeight(vTrainingLayer, vTrainingWeight, vTrainingActLayer);
			}

			//если итерации скрытых слоев
			if (s > 0 && s < numberHiddenLayer)
			{
				iWeightHiddenMin = sizeHiddenLayer * (s - 1); //первая строка вектора весов для текущей итерации
				iWeightHiddenMax = sizeHiddenLayer * s; //последняя строка вектора весов для текущей итерации

				//cout << "Первая строка итератора: " << iWeightHiddenMin << endl;
				//cout << "Последняя строка итератора: " << iWeightHiddenMax << endl << endl;

				//достает нейроны из двумерного вектора нейронов скрытого слоя
				vTrainingLayer.resize(sizeHiddenLayer); //количество нейронов одного скрытого слоя
				int count = 0; //счетчик
				for (int i = s; i < s + 1; i++) //определенная строка s 
				{
					for (int j = 0; j < sizeHiddenLayer; j++) //столбцы от 0 до 8
					{
						vTrainingLayer[count] = vHiddenLayer[i][j];
						//cout << vTrainingLayer[count] << " ";
						count++;
					}
					//cout << endl;
				}
				//cout << "Вектор правых нейронов скрытого слоя готов к обучению!" << endl << endl;

				//достает веса из двумерного вектора весов скрытого слоя 
				vTrainingWeight.resize(sizeHiddenLayer * sizeHiddenLayer); //количество весов между двумя скрытыми слоями
				count = 0;
				for (int i = sizeHiddenLayer * (s - 1); i < sizeHiddenLayer * s; i++) //от sizeHiddenLayer-1 до sizeHiddenLayer
				{
					for (int j = 0; j < sizeHiddenLayer; j++) //столбцы весов от 0 до 8
					{
						vTrainingWeight[count] = vWeightHiddenLayer[i][j];
						//cout << vTrainingWeight[count] << " ";
						count++;
					}
				/*	cout << endl;*/
				}
	/*			cout << "Вектор весов скрытого слоя готов к обучению!" << endl << endl;*/

				//достает активированные нейроны скрытого слоя 
				vTrainingActLayer.resize(sizeHiddenLayer);
				count = 0; //счетчик
				for (int i = sizeHiddenLayer * (s - 1); i < sizeHiddenLayer * s; i++) //от sizeHiddenLayer-1 до sizeHiddenLayer
				{
					vTrainingActLayer[count] = vAllActLayer[i];
		/*			cout << vTrainingActLayer[count] << " ";*/
					count++;
				}
		/*		cout << endl << "Вектор левых активированных нейронов скрытого слоя готов к обучению!" << endl << endl;*/

				//ОБУЧЕНИЕ: передаем скрытые нейроны и веса, и буфер хранения скрытых весов
				trainingHiddenWeight(vTrainingLayer, vTrainingWeight, vTrainingActLayer);
			}

			//если входная итерация
			if (s == 0)
			{
				vTrainingLayer.resize(sizeInLayer);
				for (int i = 0; i < vInLayer.size(); i++)
				{
					vTrainingLayer[i] = vInLayer[i];
		/*			cout << vTrainingLayer[i] << " ";*/
				}
	/*			cout << endl << "Вектор правых нейронов входного слоя готов к обучению!" << endl << endl;*/

				vTrainingWeight.resize(sizeInLayer*sizeHiddenLayer);
				int count = 0;

				//достает веса и нейроны входного слоя для корректировки входных нейронов
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
	/*			cout << "Вектор весов входного слоя готов к обучению!" << endl << endl;*/

				//ОБУЧЕНИЕ: передаем скрытые нейроны и веса, и буфер хранения скрытых весов
				trainingInWeight(vTrainingLayer, vTrainingWeight, vTrainingActLayer);
			}
			s -= 1;
		}
	}

	//1.тренировка выходноых весов
	void trainingOutWeight(vector<double> vTrainingLayer, vector<double> vTrainingWeight, vector<double> vTrainingActLayer)
	{
		//1.для выходных нейронов расчет ошибок между выходным и эталонным значением
		vErrLayer.resize(sizeOutLayer);

		for (int i = 0; i < vErrLayer.size(); i++) //ошибок столько сколько и выходов
		{
			vErrLayer[i] = vTrainingActLayer[i] - vConcretValues[i]; //вектор активированных нейронов минус эталонны
			//cout << vErrLayer[i] << " ";
		}
		//cout << endl << "Расчет ошибок весов выходного слоя прошел успешно!" << endl;

		//3.расчет сдвига для выходных весов
		vDeltaWeight.resize(vErrLayer.size());

		for (int i = 0; i < vDeltaWeight.size(); i++) //дельт столько сколько и ошибок
		{
			vDeltaWeight[i] = vErrLayer[i] * sigmoidDx(vTrainingActLayer[i]); //сигмоида от активированных нейронов
			//cout << vDeltaWeight[i] << " ";
		}
		//cout << endl << "Расчет сдвигов весов выходного слоя прошел успешно!" << endl << endl;

		//4.корректировка вых. весов(вес правого слоя - значение прошлого активированного нейрона * дельта весов * коэффициент обучения)
		for (int i = 0; i < sizeHiddenLayer; i++) //строка количество скрытых нейронов
		{
			for (int j = 0; j < sizeOutLayer; j++)  //столбец количетво дельт = количество выходных нейронов
			{
				vWeightOutLayer[i][j] = vWeightOutLayer[i][j] - vTrainingActLayer[i] * vDeltaWeight[j] * learningRate;
				//cout << vWeightOutLayer[i][j] << " ";
			}
			//cout << endl;
		}
		//cout << "Корректировка выходных весов прошла успешно!" << endl << endl;
	}

	//2.тренировка весов скрытого слоя
	void trainingHiddenWeight(vector<double> vTrainingLayer, vector<double> vTrainingWeight, vector<double> vTrainingActLayer)
	{
		int counter = 0;

		//проверка на первый расчет скрытого слоя
		if (s == numberHiddenLayer - 1)
		{
			cout << "Выходной метод обучения скрытых весов: " << endl << endl;
			//1.произведение новых выходных весов и дельт выходных высов
			vMultiErr.resize(sizeOutLayer*sizeHiddenLayer);

			//произведение новых весов выходного слоя и дельт выходного слоя
			for (int i = 0; i < sizeHiddenLayer; i++) //количество скрытых слоев 
			{
				for (int j = 0; j < vDeltaWeight.size(); j++) //количетсво дельт
				{
					vMultiErr[counter] = vWeightOutLayer[i][j] * vDeltaWeight[j];
					//cout << vMultiErr[counter] << " ";
					counter++;
				}
				//cout << endl;
			}
			//cout << "Произведений новых выходных весов и дельт выходных весов: " << vMultiErr.size() << endl << endl;
		}
		else
		{
			cout << "Скрытый метод обучения скрытых весов: " << endl << endl;
			//1.произведение новых выходных весов и дельт выходных высов 
			vMultiErr.resize(sizeHiddenLayer * sizeHiddenLayer);

			//произведение новых весов выходного слоя и дельт выходного слоя 
			for (int i = iWeightHiddenMin; i < iWeightHiddenMax; i++) //строки количество скрытых слоев 
			{
				for (int j = 0; j < sizeHiddenLayer; j++) //столбцы количетсво дельт = количеству нейронов скрытого слоя
				{
					vMultiErr[counter] = vWeightHiddenLayer[i][j] * vDeltaWeight[j];
					//cout << vMultiErr[counter] << " ";
					counter++;
				}
		/*		cout << endl;*/
			}
			//cout << "Произведений новых скрытых весов и дельт скрытых весов: " << vMultiErr.size() << endl << endl;
		}

		//2.Ошибки скрытого слоя = суммам произведений
		vSumErr.resize(sizeHiddenLayer); //вектор скрытых весов для одной итерации
		counter = 0;

		//строки от 0 до количества нейронов скрытого слоя(количество ошибок)
		for (int i = 0; i < sizeHiddenLayer; i++)
		{
			//столбцы - от 0 до количество скрытых нейронов(количество сумм)
			for (int j = 0; j < sizeHiddenLayer; j++)
			{
				vSumErr[i] += vMultiErr[counter];
			}
	/*		cout << vSumErr[i] << " ";*/
		}
		//cout << endl << "Количество ошибок скрытого слоя: " << vSumErr.size() << endl << endl;

		//3.расчет сдвига для весов скрытого слоя, ошибку умножает на производную сигмоиды(значения нейронов выходного слоя)
		vDeltaWeight.resize(sizeHiddenLayer); //(количество нейронов в одном скрытом слое)

		for (int i = 0; i < vDeltaWeight.size(); i++) //(количество нейронов скрытого слоя)
		{
			vDeltaWeight[i] = vSumErr[i] * sigmoidDx(vTrainingActLayer[i]); //сигмоида от активированных нейронов
			//cout << vDeltaWeight[i] << " ";
		}
		//cout << endl << "Количество дельт скрытого слоя: " << vDeltaWeight.size() << endl << endl;

		//4.корректировка весов(вес скрытого слоя - значение прошлого активированного нейрона * дельта весов * коэффициент обучения)
		vTrainingBufWeight.resize(sizeHiddenLayer*sizeHiddenLayer);
		counter = 0;
		for (int i = 0; i < sizeHiddenLayer; i++) //количество дельт
		{
			for (int j = 0; j < sizeHiddenLayer; j++) //количество активированных нейронов
			{
				//новые веса помащается в буфер весов
				vTrainingBufWeight[counter] = vTrainingWeight[counter] - vTrainingActLayer[j] * vDeltaWeight[i] * learningRate;
				counter++;
			}
		}
		//из буфера весов, сохраняет в двумерный вектор весов в правильном порядке
		counter = 0;
		for (int i = iWeightHiddenMin; i < iWeightHiddenMax; i++) //от первой до последней строки итерации скрытого слоя
		{
			for (int j = 0; j < sizeHiddenLayer; j++) //количетсов нейронов скрытого слоя
			{
				vWeightHiddenLayer[i][j] = vTrainingBufWeight[counter];
				//cout << vWeightHiddenLayer[i][j] << " ";
				counter++;
			}
			cout << endl;
		}
		//cout << "Корректировка весов скрытого слоя прошла успешно!" << endl << endl;
	}

	//3.тренировка весов входного слоя
	void trainingInWeight(vector<double> vTrainingLayer, vector<double> vTrainingWeight, vector<double> vTrainingActLayer)
	{
		//1.произведение новых весов и дельт скрытого слоя
		vMultiErr.resize(sizeInLayer*sizeHiddenLayer);
		int counter = 0;
		for (int i = 0; i < sizeInLayer; i++) //количество нейронов входного слоя = количество сумм в строке
		{
			for (int j = 0; j < vDeltaWeight.size(); j++) //количетсво дельт = количеству нейронов в скрытом слою
			{
				vMultiErr[counter] = vWeightInLayer[i][j] * vDeltaWeight[j];
				//cout << vMultiErr[counter] << " ";
				counter++;
			}
			//cout << endl;
		}
		//cout << "Произведений новых скрытых весов и дельт скрытых весов: " << vMultiErr.size() << endl << endl;

		//2.Ошибки скрытого слоя = суммам произведений
		vSumErr.resize(sizeHiddenLayer); //вектор скрытых весов для одной итерации
		counter = 0;

		//строки от 0 до количества нейронов скрытого слоя(количество ошибок)
		for (int i = 0; i < sizeHiddenLayer; i++)
		{
			//столбцы - от 0 до количество скрытых нейронов(количество сумм)
			for (int j = 0; j < sizeHiddenLayer; j++)
			{
				vSumErr[i] += vMultiErr[counter];
			}
			//cout << vSumErr[i] << " ";
		}
		//cout << endl << "Количество ошибок входного слоя: " << vSumErr.size() << endl << endl;

		//3.расчет сдвига для весов скрытого слоя, ошибку умножает на производную сигмоиды(значения нейронов выходного слоя)
		vDeltaWeight.resize(sizeHiddenLayer); //(количество нейронов в одном скрытом слое)

		for (int i = 0; i < vDeltaWeight.size(); i++) //(количество нейронов скрытого слоя)
		{
			vDeltaWeight[i] = vSumErr[i] * sigmoidDx(vTrainingActLayer[i]); //сигмоида активированного нейрона 1 скрытого слоя
			//cout << vDeltaWeight[i] << " ";
		}
		//cout << endl << "Количество дельт входного слоя: " << vDeltaWeight.size() << endl << endl;

		//4.корректировка весов(вес скрытого слоя - значение входного нейрона * дельта весов * коэффициент обучения)
		vTrainingBufWeight.resize(sizeInLayer*sizeHiddenLayer);
		counter = 0;
		for (int i = 0; i < sizeHiddenLayer; i++) //количество дельт
		{
			for (int j = 0; j < sizeInLayer; j++) //количество входных нейронов
			{
				vTrainingBufWeight[counter] = vTrainingWeight[counter] - vInLayer[j] * vDeltaWeight[i] * learningRate;
				counter++;
			}
		}
		//5.из буфера весов, сохраняет в двумерный вектор весов в правильном порядке
		counter = 0;
		for (int i = 0; i < sizeInLayer; i++) //от первой до последней строки итерации скрытого слоя
		{
			for (int j = 0; j < sizeHiddenLayer; j++) //количетсов нейронов скрытого слоя
			{
				vWeightInLayer[i][j] = vTrainingBufWeight[counter];
				//cout << vWeightInLayer[i][j] << " ";
				counter++;
			}
			//cout << endl;
		}
		//cout << "Корректировка весов входного слоя прошла успешно!" << endl << endl;
	}

};
