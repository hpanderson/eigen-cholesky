#include <iostream>
#include <chrono>

#ifdef _MSC_VER
	#pragma warning(disable:4714) // force_inline warning from eigen, could probably disable this one globally
#endif
#include <Eigen/Dense>
#include "armadillo"

class MatrixTimingFixture
{
public:
	MatrixTimingFixture() {}
	~MatrixTimingFixture() {}

	void TimeMatrixSizes();

protected:
	enum ImpType
	{
		Arma,
		Eigen
	};

	void RandomizeMembers(size_t inMatrixSize, ImpType inType);
	void DestroyMembers();
	void TimeOp(int inIterations, size_t inMatrixSize, std::string inDescription, ImpType inType, std::function<void ()> inOperation);
	std::string ImpTypeName(ImpType inType) {
		switch(inType) {
			case Arma: return "arma";
			case Eigen: return "eigen";
		}
		return "idk";
	}

	size_t mMatrixSize;
	arma::mat mArmaMatrix;
	arma::mat mArmaSymmetric;
	arma::mat mArmaTriangular;
	arma::mat mAnotherArma;
	arma::mat mArmaUneven;
	arma::vec mArmaVector;
	arma::mat mXtXArma;

	Eigen::MatrixXd mEigenMatrix;
	Eigen::MatrixXd mEigenSymmetric;
	Eigen::MatrixXd mEigenTriangular;
	Eigen::MatrixXd mAnotherEigen;
	Eigen::MatrixXd mEigenUneven;
	Eigen::VectorXd mEigenVector;
	Eigen::MatrixXd mXtXEigen;
};

void MatrixTimingFixture::RandomizeMembers(size_t inMatrixSize, MatrixTimingFixture::ImpType inType)
{
	switch (inType)
	{
		case Arma:
		{
			mArmaMatrix = arma::mat((arma::uword)inMatrixSize, (arma::uword)inMatrixSize, arma::fill::randn);
			mAnotherArma = arma::mat((arma::uword)inMatrixSize, (arma::uword)inMatrixSize, arma::fill::randn);
			mArmaVector = arma::vec((arma::uword)inMatrixSize, arma::fill::randn);
			mArmaUneven = arma::mat((arma::uword)inMatrixSize, (arma::uword)inMatrixSize * 2, arma::fill::randn);

			mArmaTriangular = arma::trimatu(arma::mat((arma::uword)inMatrixSize, (arma::uword)inMatrixSize, arma::fill::randn));
			mArmaSymmetric = arma::symmatu(arma::mat((arma::uword)inMatrixSize, (arma::uword)inMatrixSize, arma::fill::randn));
			mXtXArma = mArmaMatrix.t() * mArmaMatrix;
			break;
		}

		case Eigen:
		{
			mEigenMatrix = Eigen::MatrixXd::Random(inMatrixSize, inMatrixSize);
			mAnotherEigen = Eigen::MatrixXd::Random(inMatrixSize, inMatrixSize);
			mEigenVector = Eigen::VectorXd::Random(inMatrixSize);
			mEigenUneven = Eigen::MatrixXd::Random(inMatrixSize, inMatrixSize * 2);

			mEigenTriangular = Eigen::MatrixXd::Random(inMatrixSize, inMatrixSize);
			mEigenSymmetric = Eigen::MatrixXd::Random(inMatrixSize, inMatrixSize);
			mXtXEigen = mEigenMatrix.transpose() * mEigenMatrix;
			break;
		}
	}
}

void MatrixTimingFixture::DestroyMembers()
{
	mArmaMatrix.clear();
	mXtXArma.clear();

	// no clear function?
	mEigenMatrix.resize(0, 0);
	mXtXEigen.resize(0, 0);
}

void MatrixTimingFixture::TimeOp(int inIterations, size_t inMatrixSize, std::string inDescription, MatrixTimingFixture::ImpType inType, std::function<void ()> inOperation)
{
	using namespace std;
	using namespace chrono;

	nanoseconds totalTime;
	for (int i = 0; i < inIterations; ++i)
	{
		// fill with random values each iteration
		RandomizeMembers(inMatrixSize, inType);

		auto start = high_resolution_clock::now();
		inOperation();
		totalTime += high_resolution_clock::now() - start;

		DestroyMembers();
	}

	auto us = totalTime.count() / 1000.0;
	std::cout << "Average time for " << inDescription << "_" << ImpTypeName(inType) << " (" << inMatrixSize << " elements, " << inIterations << " iterations): " << (us / (double)inIterations) << " us" << std::endl;
}


void MatrixTimingFixture::TimeMatrixSizes()
{
	int iterations = 10;
	int maxPow = 10;
	// go through various matrix sizes and time several BLAS/LAPACK functions
	for (int b = 1; b <= maxPow; ++b)
	{
		size_t matrixSize = (size_t)pow(2, b);

		TimeOp(iterations, matrixSize, "qr", Arma, [&]() {
			arma::mat R;
			arma::mat Q;
			arma::qr(Q, R, mArmaMatrix);
		});

		TimeOp(iterations, matrixSize, "qr", Eigen, [&]() {
			Eigen::MatrixXd qr = mEigenMatrix.householderQr().matrixQR();
		});

		TimeOp(iterations, matrixSize, "cholesky", Arma, [&]() {
			arma::mat R = arma::chol(mXtXArma);
		});

		TimeOp(iterations, matrixSize, "cholesky", Eigen, [&]() {
			Eigen::LLT<Eigen::MatrixXd> chol(mXtXEigen);
		});

		TimeOp(iterations, matrixSize, "trinv", Arma, [&]() {
			arma::mat inverse = arma::inv(arma::trimatu(mArmaTriangular));
		});

		TimeOp(iterations, matrixSize, "trinv", Eigen, [&]() {
			Eigen::MatrixXd inverse = mEigenTriangular.inverse();
		});

		TimeOp(iterations, matrixSize, "gemm", Arma, [&]() {
			arma::mat result = mArmaMatrix * mAnotherArma;
		});

		TimeOp(iterations, matrixSize, "gemm", Eigen, [&]() {
			Eigen::MatrixXd result = mEigenMatrix * mAnotherEigen;
		});

		TimeOp(iterations, matrixSize, "symm", Arma, [&]() {
			arma::mat result = mArmaSymmetric * mAnotherArma;
		});

		TimeOp(iterations, matrixSize, "symm", Eigen, [&]() {
			Eigen::MatrixXd result = mEigenSymmetric * mAnotherEigen;
		});

		TimeOp(iterations, matrixSize, "symmt", Arma, [&]() {
			arma::mat result = mArmaSymmetric * mAnotherArma.t();
		});

		TimeOp(iterations, matrixSize, "symmt", Eigen, [&]() {
			Eigen::MatrixXd result = mEigenSymmetric * mAnotherEigen.transpose();
		});

		TimeOp(iterations, matrixSize, "trimm", Arma, [&]() {
			arma::mat result = mArmaTriangular * mAnotherArma;
		});

		TimeOp(iterations, matrixSize, "trimm", Eigen, [&]() {
			Eigen::MatrixXd result = mEigenTriangular * mAnotherEigen;
		});

		TimeOp(iterations, matrixSize, "crossprod", Arma, [&]() {
			arma::mat result = mArmaMatrix * mArmaMatrix.t();
		});

		TimeOp(iterations, matrixSize, "crossprod", Eigen, [&]() {
			Eigen::MatrixXd result = mEigenMatrix * mEigenMatrix.transpose();
		});

		TimeOp(iterations, matrixSize, "gemv", Arma, [&]() {
			arma::vec result = mArmaMatrix * mArmaVector;
		});

		TimeOp(iterations, matrixSize, "gemv", Eigen, [&]() {
			Eigen::VectorXd result = mEigenMatrix * mEigenVector;
		});

		TimeOp(iterations, matrixSize, "symv", Arma, [&]() {
			arma::vec result = mArmaSymmetric * mArmaVector;
		});

		TimeOp(iterations, matrixSize, "symv", Eigen, [&]() {
			Eigen::VectorXd result = mEigenSymmetric * mEigenVector;
		});

		TimeOp(iterations, matrixSize, "trimv", Arma, [&]() {
			arma::vec result = mArmaTriangular * mArmaVector;
		});

		TimeOp(iterations, matrixSize, "trimv", Eigen, [&]() {
			Eigen::VectorXd result = mEigenTriangular * mEigenVector;
		});

		TimeOp(iterations, matrixSize, "XtX", Arma, [&]() {
			arma::mat result = mArmaUneven.t() * mArmaUneven;
		});

		TimeOp(iterations, matrixSize, "XtX", Eigen, [&]() {
			Eigen::MatrixXd result = mEigenUneven.transpose() * mEigenUneven;
		});

		TimeOp(iterations, matrixSize, "gescale", Arma, [&]() {
			mArmaMatrix *= 2;
		});

		TimeOp(iterations, matrixSize, "gescale", Eigen, [&]() {
			mEigenMatrix *= 2;
		});

		TimeOp(iterations, matrixSize, "trscale", Arma, [&]() {
			mArmaTriangular *= 2;
		});

		TimeOp(iterations, matrixSize, "trscale", Eigen, [&]() {
			mEigenTriangular *= 2;
		});

		TimeOp(iterations, matrixSize, "syscale", Arma, [&]() {
			mArmaSymmetric *= 2;
		});

		TimeOp(iterations, matrixSize, "syscale", Eigen, [&]() {
			mEigenSymmetric *= 2;
		});
	}
}

int main(int inArgCount, char* inArgs[])
{
	MatrixTimingFixture mtf;
	mtf.TimeMatrixSizes();
}

