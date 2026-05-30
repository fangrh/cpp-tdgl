#pragma once

#include "device/device.h"
#include "mesh/mesh.h"
#include <Eigen/Core>
#include <highfive/H5File.hpp>
#include <string>
#include <vector>

class JeSolutionWriter {
public:
    JeSolutionWriter(const std::string& run_dir, const Mesh& mesh, const Device& device);
    ~JeSolutionWriter() = default;

    void write_result(double je_value,
                     const std::vector<Eigen::VectorXcd>& psi_history,
                     const std::vector<Eigen::MatrixX2d>& vector_potential_history,
                     const std::vector<Eigen::MatrixX2d>& superfluid_velocity_history,
                     const std::vector<Eigen::MatrixX2d>& electric_field_history,
                     const std::vector<Eigen::MatrixX2d>& current_history,
                     const std::string& run_id,
                     const std::string& solver = "cpp-tdgl",
                     int completed_steps = 0);

    static std::string je_filename(double je_value);

private:
    std::string run_dir_;
    const Mesh& mesh_;
    const Device& device_;

    void write_atomic(const std::string& target_path, double je_value,
                     const std::vector<Eigen::VectorXcd>& psi_history,
                     const std::vector<Eigen::MatrixX2d>& vector_potential_history,
                     const std::vector<Eigen::MatrixX2d>& superfluid_velocity_history,
                     const std::vector<Eigen::MatrixX2d>& electric_field_history,
                     const std::vector<Eigen::MatrixX2d>& current_history,
                     const std::string& run_id,
                     const std::string& solver,
                     int completed_steps);
};

// Initial state loader for per-Je simulations
Eigen::VectorXcd load_initial_psi(const std::string& h5_path);
