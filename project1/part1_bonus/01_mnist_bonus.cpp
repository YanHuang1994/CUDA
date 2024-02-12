#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <cmath> // For std::exp and std::log functions

int main() {
    // m 和 b 的初始值
    double m = -2; 
    double b = 40;

    // 样本数据
    Eigen::VectorXd x(10);
    Eigen::VectorXd y(10);
    x << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9;
    y << 10, 20, 25, 30, 40, 45, 40, 50, 60, 55;
    
    // 计算 y_hat
    Eigen::VectorXd y_hat = x * m + Eigen::VectorXd::Ones(x.size()) * b;

    // 计算损失
    double loss = (y - y_hat).array().square().sum() / x.size();

    std::cout << "Loss: " << loss << std::endl;

    return 0;
}
