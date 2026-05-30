#include "solution/je_solution.h"
#include <highfive/H5File.hpp>
#include <highfive/H5DataSet.hpp>
#include <filesystem>
#include <stdexcept>
#include <cstdio>

namespace h5 = HighFive;

template<typename T>
void write_1d(h5::Group& g, const std::string& name,
              const T* data, size_t n) {
    h5::DataSpace space({n});
    g.createDataSet(name, space, h5::AtomicType<T>()).write_raw(data);
}

template<typename T>
void write_2d(h5::Group& g, const std::string& name,
              const T* data, size_t rows, size_t cols) {
    h5::DataSpace space({rows, cols});
    g.createDataSet(name, space, h5::AtomicType<T>()).write_raw(data);
}

template<typename T>
void write_3d(h5::Group& g, const std::string& name,
              const T* data, size_t d1, size_t d2, size_t d3) {
    h5::DataSpace space({d1, d2, d3});
    g.createDataSet(name, space, h5::AtomicType<T>()).write_raw(data);
}

JeSolutionWriter::JeSolutionWriter(const std::string& run_dir, const Mesh& mesh, const Device& device)
    : run_dir_(run_dir), mesh_(mesh), device_(device) {
    // Create run directory if it doesn't exist
    if (!std::filesystem::exists(run_dir_)) {
        std::filesystem::create_directories(run_dir_);
    }
}

std::string JeSolutionWriter::je_filename(double je_value) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "result_Je_%010.2f.h5", je_value);
    return std::string(buf);
}

void JeSolutionWriter::write_result(double je_value,
                                   const std::vector<Eigen::VectorXcd>& psi_history,
                                   const std::vector<Eigen::MatrixX2d>& vector_potential_history,
                                   const std::vector<Eigen::MatrixX2d>& superfluid_velocity_history,
                                   const std::vector<Eigen::MatrixX2d>& electric_field_history,
                                   const std::vector<Eigen::MatrixX2d>& current_history,
                                   const std::string& run_id,
                                   const std::string& solver,
                                   int completed_steps) {
    std::string target_path = run_dir_ + "/" + je_filename(je_value);
    write_atomic(target_path, je_value, psi_history, vector_potential_history,
                superfluid_velocity_history, electric_field_history, current_history,
                run_id, solver, completed_steps);
}

void JeSolutionWriter::write_atomic(const std::string& target_path, double je_value,
                                   const std::vector<Eigen::VectorXcd>& psi_history,
                                   const std::vector<Eigen::MatrixX2d>& vector_potential_history,
                                   const std::vector<Eigen::MatrixX2d>& superfluid_velocity_history,
                                   const std::vector<Eigen::MatrixX2d>& electric_field_history,
                                   const std::vector<Eigen::MatrixX2d>& current_history,
                                   const std::string& run_id,
                                   const std::string& solver,
                                   int completed_steps) {
    // Write to .tmp file first, then rename for atomicity
    std::string tmp_path = target_path + ".tmp";

    try {
        h5::File file(tmp_path, h5::File::Overwrite);

        // Write attributes
        file.createAttribute("Je", je_value).write(je_value);
        file.createAttribute("run_id", run_id).write(run_id);
        file.createAttribute("solver", solver).write(solver);
        file.createAttribute("completed_steps", completed_steps).write(completed_steps);

        // Check that all history vectors have the same size
        size_t n_steps = psi_history.size();
        if (vector_potential_history.size() != n_steps ||
            superfluid_velocity_history.size() != n_steps ||
            electric_field_history.size() != n_steps ||
            current_history.size() != n_steps) {
            throw std::runtime_error("History vectors must all have the same size");
        }

        if (n_steps == 0) {
            throw std::runtime_error("Cannot write empty history");
        }

        int n_sites = psi_history[0].size();
        int n_edges = vector_potential_history[0].rows();

        // Prepare data arrays with shape (n_steps, n_sites, 2) for 2D fields
        // and (n_steps, n_sites) for psi
        std::vector<double> psi_data(n_steps * n_sites * 2);
        std::vector<double> vector_potential_data(n_steps * n_edges * 2);
        std::vector<double> superfluid_velocity_data(n_steps * n_edges * 2);
        std::vector<double> electric_field_data(n_steps * n_edges * 2);
        std::vector<double> current_data(n_steps * n_edges * 2);

        for (size_t t = 0; t < n_steps; ++t) {
            // Write psi as (n_steps, n_sites, 2) complex (interleaved real/imag)
            for (int i = 0; i < n_sites; ++i) {
                psi_data[t * n_sites * 2 + i * 2] = psi_history[t](i).real();
                psi_data[t * n_sites * 2 + i * 2 + 1] = psi_history[t](i).imag();
            }

            // Write 2D fields as (n_steps, n_edges, 2)
            for (int i = 0; i < n_edges; ++i) {
                vector_potential_data[t * n_edges * 2 + i * 2] = vector_potential_history[t](i, 0);
                vector_potential_data[t * n_edges * 2 + i * 2 + 1] = vector_potential_history[t](i, 1);

                superfluid_velocity_data[t * n_edges * 2 + i * 2] = superfluid_velocity_history[t](i, 0);
                superfluid_velocity_data[t * n_edges * 2 + i * 2 + 1] = superfluid_velocity_history[t](i, 1);

                electric_field_data[t * n_edges * 2 + i * 2] = electric_field_history[t](i, 0);
                electric_field_data[t * n_edges * 2 + i * 2 + 1] = electric_field_history[t](i, 1);

                current_data[t * n_edges * 2 + i * 2] = current_history[t](i, 0);
                current_data[t * n_edges * 2 + i * 2 + 1] = current_history[t](i, 1);
            }
        }

        // Write datasets
        // Note: py-tdgl stores psi as (N_times, N_sites) complex, but HDF5 complex is not
        // universally supported. We store as (N_times, N_sites, 2) float64 to match.
        h5::DataSpace psi_space({n_steps, static_cast<size_t>(n_sites), 2ULL});
        file.createDataSet<double>("psi", psi_space).write_raw(psi_data.data());

        h5::DataSpace va_space({n_steps, static_cast<size_t>(n_edges), 2ULL});
        file.createDataSet<double>("vector_potential", va_space).write_raw(vector_potential_data.data());
        file.createDataSet<double>("superfluid_velocity", va_space).write_raw(superfluid_velocity_data.data());
        file.createDataSet<double>("electric_field", va_space).write_raw(electric_field_data.data());
        file.createDataSet<double>("current", va_space).write_raw(current_data.data());

        file.flush();

    } catch (...) {
        // Clean up .tmp file on error
        if (std::filesystem::exists(tmp_path)) {
            std::filesystem::remove(tmp_path);
        }
        throw;
    }

    // Atomic rename
    std::filesystem::rename(tmp_path, target_path);
}

// Initial state loader
Eigen::VectorXcd load_initial_psi(const std::string& h5_path) {
    h5::File file(h5_path, h5::File::ReadOnly);

    // Read psi dataset
    auto psi_ds = file.getDataSet("psi");
    auto psi_space = psi_ds.getSpace();
    auto dims = psi_space.getDimensions();

    if (dims.size() != 3) {
        throw std::runtime_error("Expected psi dataset to have 3 dimensions (n_steps, n_sites, 2)");
    }

    size_t n_steps = dims[0];
    size_t n_sites = dims[1];

    // Read entire psi array
    std::vector<double> psi_data(n_steps * n_sites * 2);
    psi_ds.read(psi_data.data());

    // Extract the last time step
    Eigen::VectorXcd psi(n_sites);
    for (size_t i = 0; i < n_sites; ++i) {
        psi(i) = std::complex<double>(
            psi_data[(n_steps - 1) * n_sites * 2 + i * 2],
            psi_data[(n_steps - 1) * n_sites * 2 + i * 2 + 1]
        );
    }

    return psi;
}
