#include <Eigen/Dense>
#include "SpherePlanning.h"
#include <iostream>
#include <fstream>

int main(int argc, char **argv)
{
    ompl::msg::setLogLevel(ompl::msg::LOG_DEBUG);
    SpherePlanning spherePlanning;
    Eigen::Vector3d start, goal;
    srand(17);

    std::ofstream output_csv("./data/stat_CoMPNet.csv");
    output_csv << "no, time" << std::endl;

    for (unsigned int i = 0; i < 1000; i++)
    {
        std::cout << "Sample " << i << std::endl;
        do
        {
            while (true)
            {
                start.setRandom();
                start.normalize();
                if (spherePlanning.isValid(start))
                    break;
            }
            while (true)
            {
                goal.setRandom();
                goal.normalize();
                if (spherePlanning.isValid(goal))
                    break;
            }
        } while ((start - goal).norm() < 1.9);

        bool solved = spherePlanning.planOnce(start, goal);
        if (solved)
        {
            output_csv << i << ", " << spherePlanning.getTime() << std::endl;
        }
        else {
            output_csv << i << ", " << "inf" << std::endl;
        }
        spherePlanning.clear();
    }
    output_csv.close();
}