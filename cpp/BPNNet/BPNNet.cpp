#include "BPNNet.h"
#include <random>
#include <math.h>

//以下是测试时用到的库，测试完毕后需要删除
#include <iostream>



using namespace std;

clock_t timeStart, timeEnd;
BPLayer::BPLayer(int nodeNum, int nodeBef, LAYERTYPE layerType, ActivationFun::ActivationFunType type) {
	isInitial = false;
	SetActivationFun(type);
	this->layerType = layerType;
	this->nodeNum = nodeNum;
	this->nodeBef = nodeBef;


	this->nodeVal = new double[nodeNum];
	memset(this->nodeVal, 0, sizeof(double) * nodeNum);

	this->nodeValUnActived = new double[nodeNum];
	memset(this->nodeValUnActived, 0, sizeof(double) * nodeNum);

	this->tempNodeValForeProgation = new double[nodeNum];
	memset(this->tempNodeValForeProgation, 0, sizeof(double) * nodeNum);


	this->bias = new double[nodeNum];
	memset(this->bias, 0, sizeof(double) * nodeNum);

	this->biasBest = new double[nodeNum];
	memset(this->biasBest, 0, sizeof(double) * nodeNum);

	this->weight = new double[nodeNum * nodeBef];
	memset(weight, 0, nodeNum * nodeBef * sizeof(double));

	this->weightBest = new double[nodeNum * nodeBef];
	memset(weightBest, 0, nodeNum * nodeBef * sizeof(double));

	this->dWeight = new double[nodeBef];
	memset(dWeight, 0, nodeBef * sizeof(double));

	this->tempDWeightBackProgation = new double[nodeBef];
	memset(tempDWeightBackProgation, 0, nodeBef * sizeof(double));

	this->diffActivationFun = new double[nodeNum];
	memset(diffActivationFun, 0, sizeof(double) * nodeNum);

	this->xBack = new double[nodeNum];
	memset(xBack, 0, sizeof(double) * nodeNum);

	this->tempXBackBackProgation = new double[nodeNum];
	memset(tempXBackBackProgation, 0, sizeof(double) * nodeNum);

	isInitial = false;
}

bool BPLayer::Initial(default_random_engine e)
{
	static uniform_real_distribution<double> u(-1, 1);
	for (int i = 0; i < nodeNum * nodeBef; i++) {
		weightBest[i] = u(e);
	}
	for (int i = 0; i < nodeNum; i++) {
		biasBest[i] = u(e);
	}
	SynParameter();

	isInitial = true;
	return true;
}

bool BPLayer::Initial()
{
	static uniform_real_distribution<double> u(-0.5, 0.5);
	for (int i = 0; i < nodeNum * nodeBef; i++) {
		weightBest[i] = u(randomEngine);
	}
	for (int i = 0; i < nodeNum; i++) {
		biasBest[i] = u(randomEngine);
	}
	SynParameter();

	isInitial = true;
	return true;
}

bool BPLayer::ForePropagate(const BPLayer* layerBef)
{
	if (!isInitial)
	{
		printf("Layer not initialized\n"); 
		return false;
	}	
	if (layerBef->nodeNum != nodeBef) {
		printf("Not matched node num with last layer\n");
		return false;
	}

	MatrixMul(weightBest, layerBef->nodeVal, nodeNum, nodeBef, 1, tempNodeValForeProgation);
	MatrixAdd(tempNodeValForeProgation, biasBest, nodeNum, nodeValUnActived);
	activationFun->Active(nodeValUnActived, nodeVal, nodeNum);
	return true;
}

bool BPLayer::BackPropagate(const BPLayer* layerBef, const BPLayer* layerAft, double step)
{
	if (!isInitial)
	{
		printf("Layer not initialized\n"); 
		return false;
	}

	activationFun->Diff(nodeVal, nodeValUnActived, diffActivationFun, nodeNum);
	MatrixMul(layerAft->xBack, layerAft->weight, 1, layerAft->nodeNum, nodeNum, tempXBackBackProgation);
	MatrixDot(tempXBackBackProgation, diffActivationFun, nodeNum, xBack);
	

	for (int i = 0; i < nodeNum; i++) {


		MatrixDot(layerBef->nodeVal, xBack[i], nodeBef, tempDWeightBackProgation);
		MatrixDot(tempDWeightBackProgation, step, nodeBef, dWeight);
		MatrixSub(weight + i * nodeBef, dWeight, nodeBef, weightBest + i * nodeBef);
		biasBest[i] -= step * xBack[i];
	}
	return true;
}

bool BPLayer::BackPropagate(const BPLayer* layerBef, double step)
{
	if (!isInitial)
	{
		printf("Layer not initialized\n");
		return false;
	}
	activationFun->Diff(nodeVal, nodeValUnActived, diffActivationFun, nodeNum);

	//这里是建立在网络已经更具能量函数对xBack做了部分赋值
	MatrixDot(tempXBackBackProgation, diffActivationFun, nodeNum, xBack);

	for (int i = 0; i < nodeNum; i++) {

		MatrixDot(layerBef->nodeVal, xBack[i], nodeBef, tempDWeightBackProgation);
		MatrixDot(tempDWeightBackProgation, step, nodeBef, dWeight);
		MatrixSub(weight + i * nodeBef, dWeight, nodeBef, weightBest + i * nodeBef);
		biasBest[i] -= step * xBack[i];
	}
	return true;
}

bool BPLayer::SetActivationFun(ActivationFun::ActivationFunType type)
{
	switch (type)
	{
	case ActivationFun::SIGMOID:
		activationFun = new ActivationFunSigmoid;
		break;
	default:
		activationFun = new ActivationFunSigmoid;
		break;
	}
	return true;
}

void BPLayer::SetTempXBack(double* xBack)
{
	memcpy(this->tempXBackBackProgation, xBack, sizeof(double) * nodeNum);
}

void BPLayer::SynParameter()
{
	//memcpy更为高效
	memcpy(weight, weightBest, sizeof(double) * nodeBef * nodeNum);
	memcpy(bias, biasBest, sizeof(double) * nodeNum);
}



BPLayer::~BPLayer() {
	delete nodeVal;
	delete tempNodeValForeProgation;
	delete nodeValUnActived;
	delete bias;
	delete biasBest;
	delete weight;
	delete weightBest;
	delete dWeight;
	delete diffActivationFun;
	delete xBack;
	delete activationFun;
}

BPNNet::BPNNet(int layerNum, int* nodeNums, EnergyFun::EnergyFunType type)
{
	//randomEngine.seed();

	step = STEP;
	isInitial = false;
	SetEnergyFun(type);
	layerNumHiddenLayer = layerNum - 2;
	if (layerNumHiddenLayer < 0) {
		printf("Layer num must larger than 1\n");
		throw BPNNetException("Wrong layer num set");
	}

	nodeNumHiddenLayer = new int[layerNum - 2];
	memset(nodeNumHiddenLayer, 0, sizeof(int) * (layerNum - 2));
	try {
		for (int i = 0; i < layerNum; i++) {
			int nodeNum = nodeNums[i];
			if (nodeNum <= 0)
				throw BPNNetException("");
			if (i == 0) {
				nodeNumInputLayer = nodeNums[i];
			}
			else if (i == layerNum - 1) {
				nodeNumOutLayer = nodeNums[i];
			}
			else {
				nodeNumHiddenLayer[i - 1] = nodeNum;
			}
		}
	}
	catch(exception e){
		throw BPNNetException("Wrong layers num array input");
	}

	net = new NetNode;
	NetNode* nowLayer = net;
	nowLayer->nodeBef = NULL;
	nowLayer->layer = new BPLayer(nodeNumInputLayer, 0, BPLayer::VITUAL);
	nowLayer->nodeNext = new NetNode;
	nowLayer->nodeNext->nodeBef = nowLayer;
	nowLayer = nowLayer->nodeNext;

	nowLayer->layer = new BPLayer(nodeNumInputLayer, nodeNumInputLayer, BPLayer::INPUT);
	nowLayer->nodeNext = new NetNode;
	nowLayer->nodeNext->nodeBef = nowLayer;
	nowLayer = nowLayer->nodeNext;
	for (int i = 0; i < layerNumHiddenLayer; i++) {
		int nodeBef, nodeNum;
		nodeNum = nodeNumHiddenLayer[i];
		nodeBef = i == 0 ? nodeNumInputLayer : nodeNumHiddenLayer[i - 1];
		nowLayer->layer = new BPLayer(nodeNum, nodeBef, BPLayer::HIDDEN);
		nowLayer->nodeNext = new NetNode;
		nowLayer->nodeNext->nodeBef = nowLayer;
		nowLayer = nowLayer->nodeNext;
	}
	nowLayer->layer = new BPLayer(nodeNumOutLayer, layerNumHiddenLayer == 0 ? nodeNumInputLayer : nodeNumHiddenLayer[layerNumHiddenLayer - 1], BPLayer::OUTPUT);
	nowLayer->nodeNext = NULL;

	outputLayer = nowLayer;

	diffEnergyFun = new double[nodeNumOutLayer];
	memset(diffEnergyFun, 0, sizeof(double) * nodeNumOutLayer);

	output = new double[nodeNumOutLayer];
	memset(output, 0, sizeof(double) * nodeNumOutLayer);
}

BPNNet::~BPNNet()
{
	//释放net中的
	delete net->layer;
	while (net->nodeNext != NULL) {
		net = net->nodeNext;
		delete net->nodeBef;
		delete net->layer;
	}
	delete net;

	delete nodeNumHiddenLayer;
	delete output;
	delete energyFun;
	delete diffEnergyFun;
}

bool BPNNet::SetActivationFun(int nLayer, ActivationFun::ActivationFunType activationFunType)
{
	NetNode* nowLayer = net->nodeNext;
	for (int i = 0; i < nLayer; i++) {
		nowLayer = nowLayer->nodeNext;
		if (nowLayer == NULL) {
			break;
		}
	}
	if (nowLayer == NULL) {
		return false;
	}
	nowLayer->layer->SetActivationFun(activationFunType);
	nowLayer = NULL;
	return true;
}

bool BPNNet::SetActivationFun(ActivationFun::ActivationFunType* activationFunType)
{
	NetNode* nowLayer = net->nodeNext;
	int count = 0;
	try {
		while (true) {
			nowLayer->layer->SetActivationFun(activationFunType[count++]);
			nowLayer = nowLayer->nodeNext;
			if (nowLayer == NULL) {
				break;
			}
		}
	}
	catch (...) {
		return false;
	}
	return true;

	
}

bool BPNNet::SetEnergyFun(EnergyFun::EnergyFunType type)
{
	switch (type)
	{
	case EnergyFun::STD:
		energyFun = new EnergyFunStd;
		break;
	default:
		energyFun = new EnergyFunStd;
		break;
	}
	return true;
}

bool BPNNet::Initial()
{
	//按net初始化
	NetNode* nowLayer = net->nodeNext;
	default_random_engine e(GetTickCount64());
	while (nowLayer != NULL) {
		if (!(nowLayer->layer->Initial()))
			return false;
		nowLayer = nowLayer->nodeNext;
	}

	isInitial = true;
	return true;
}

void BPNNet::SetStep(double step)
{
	this->step = step;
}

bool BPNNet::ForePropagate(double* input)
{
	if (!isInitial)
	{
		printf("Net not initialized\n"); 
		return false;
	}
	//用net传播
	try {
		for (int i = 0; i < nodeNumInputLayer; i++) {
			net->layer->nodeVal[i] = input[i];
		}
	}
	catch (...) {
		throw BPNNetException("Wrong input");
	}
	NetNode* nowLayer = net;
	while (nowLayer->nodeNext != NULL) {
		nowLayer = nowLayer->nodeNext;
		nowLayer->layer->ForePropagate(nowLayer->nodeBef->layer);
	}

	for (int i = 0; i < nodeNumOutLayer; i++) {
		output[i] = nowLayer->layer->nodeVal[i];
	}
	return true;
}

bool BPNNet::BackPropagate(double* target)
{
	if (!isInitial)
	{
		printf("Net not initialized\n");
		return false;
	}
	//输出层传播特殊处理

	energyFun->Diff(output, target, nodeNumOutLayer, diffEnergyFun);
	outputLayer->layer->SetTempXBack(diffEnergyFun);
	outputLayer->layer->BackPropagate(outputLayer->nodeBef->layer, step);


	//其它层传播
	NetNode* nowLayer = outputLayer->nodeBef;
	while (nowLayer->layer->layerType != BPLayer::VITUAL) {
		nowLayer->layer->BackPropagate(nowLayer->nodeBef->layer, nowLayer->nodeNext->layer, step);
		nowLayer = nowLayer->nodeBef;
	}


	//所有层用weightBest和biasBest同步weight和bias
	nowLayer = net->nodeNext;
	while (true) {
		nowLayer->layer->SynParameter();
		if (nowLayer->nodeNext == NULL)
			break;
		nowLayer = nowLayer->nodeNext;
	}

	return true;
}

bool BPNNet::Train(double* input, double* target, int dataSetSize, int cycle)
{
	static uniform_int_distribution<int> u(0, dataSetSize-1);
	int index = 0;
	for (int cyc = 0; cyc < cycle; cyc++) {
		index = u(randomEngine);
		ForePropagate(input + index * nodeNumInputLayer);
		BackPropagate(target + index * nodeNumOutLayer);

		////显示本次下降后的能量值
		//ForePropagate(input + index * nodeNumInputLayer);
		//energyFun->Fun(output, target + index * nodeNumOutLayer, nodeNumOutLayer, &energy);
		//printf("%f\n", energy);
	}
	//for (int cyc = 0; cyc < cycle; cyc++) {
	//	for (int i = 0; i < dataSetSize; i++) {
	//		ForePropagate(input + i * nodeNumInputLayer);
	//		BackPropagate(target + i * nodeNumOutLayer);
	//	}

	//	energyFun->Fun(output, target, nodeNumOutLayer, &energy);
	//	printf("%f\n", energy);
	//}

	return true;
}

void EnergyFunStd::Fun(double* x, double* target, int n, double* y)
{
	y[0] = 0;
	for (int i = 0; i < n; i++) {
		y[0] += (x[i] - target[i]) * (x[i] - target[i]);
	}
	y[0] = 0.5 * y[0];

}

void EnergyFunStd::Diff(double* x, double* target, int n, double* diff)
{
	for (int i = 0; i < n; i++) {
		diff[i] = (x[i] - target[i]);
	}
}

void ActivationFunSigmoid::Active(double* x, double* xActived, int n)
{
	for (int i = 0; i < n; i++) {
		xActived[i] = 1 / (1 + exp(-x[i]));
	}
}

void ActivationFunSigmoid::Diff(double* xActived, double* xUnActived, double* diff, int n)
{
	//诶，就是玩，传进来不用
	for (int i = 0; i < n; i++) {
		diff[i] = xActived[i] * (1 - xActived[i]);
	}
}

void MatrixDot(double* mat, double x, int size, double* result)
{
	for (int i = 0; i < size; i++)
		result[i] = mat[i] * x;
}

void MatrixDot(double* mat1, double* mat2, int size, double* result)
{
	for (int i = 0; i < size; i++)
		result[i] = mat1[i] * mat2[i];
}

void MatrixDot(double* mat, double* vector, int row, int col, bool isRowVec, double* result)
{
	if (isRowVec) {
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				result[i * col + j] = mat[i * col + j] * vector[j];
			}
		}
	}
	else {
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				result[i * col + j] = mat[i * col + j] * vector[i];
			}
		}
	}
}

void MatrixAdd(double* mat1, double* mat2, int size, double* result)
{
	for (int i = 0; i < size; i++)
		result[i] = mat1[i] + mat2[i];
}

void MatrixSub(double* mat1, double* mat2, int size, double* result)
{
	for (int i = 0; i < size; i++)
		result[i] = mat1[i] - mat2[i];
}

void MatrixMul(double* mat1, double* mat2, int row1, int col1_row2, int col2, double* result)
{
	memset(result, 0, sizeof(double) * row1 * col2);
	for (int rowIdx = 0; rowIdx < row1; rowIdx++) {
		for (int colIdx = 0; colIdx < col2; colIdx++) {
			for (int i = 0; i < col1_row2; i++) {
				result[rowIdx * col2 + colIdx] += mat1[rowIdx * col1_row2 + i] * mat2[i * col2 + colIdx];
			}
		}
	}
}

int MaxIndex(double* input, int size)
{
	int maxIndex = 0;
	for (int i = 1; i < size; i++) {
		if (input[i] > input[maxIndex])
			maxIndex = i;
	}
	return maxIndex;
}
