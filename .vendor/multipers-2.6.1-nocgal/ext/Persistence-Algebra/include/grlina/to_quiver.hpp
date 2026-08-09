#pragma once

#include <grlina/orders_and_graphs.hpp>
#include <grlina/sparse_matrix.hpp>

namespace graded_linalg {

template <typename index>
array<index> equidistant_grid_edges(int m, index n) {
    array<index> result(m * n);
    
    for(index i = 0; i < m; i++) {
        for(index j = 0; j < n; j++) {
            index idx = i * n + j;
            
            // Right neighbor (j+1)
            if(j + 1 < n) {
                result[idx].push_back(i * n + (j + 1));
            }
            
            // Top neighbor (i+1)
            if(i + 1 < m) {
                result[idx].push_back((i + 1) * n + j);
            }
        }
    }
    
    return result;
}

template <typename index, typename D>
struct QuiverRepresentation {
	vec<D> degrees;
    vec<index> dimensionVector;
	edge_list<index> edges;
	vec<SparseMatrix<index>> matrices;
    
    QuiverRepresentation(){
        degrees = vec<D>();
        dimensionVector = vec<index>();
        edges = edge_list<index>();
        matrices = vec<SparseMatrix<index>>();
    }

    void print(){
        std::cout << "Degrees: ";
        for(auto d : degrees){
            Degree_traits<D>::print_degree(d);
            std::cout << " ";
        }
        std::cout << std::endl;
        print_edge_list(edges);
        index size = degrees.size();
        assert(size == dimensionVector.size());
        index length = matrices.size();
        assert(length == edges.size());
        for(index i = 0; i < length; i++){
            std::cout << "Matrix at edge " << i << std::endl;
            matrices[i].print();
        }
    }

    QuiverRepresentation(vec<D> degrees, const vec<index>& dimensionVector, const edge_list<index>& edges, const vec<SparseMatrix<index>>& matrices) : 
    degrees(degrees), dimensionVector(dimensionVector), edges(edges), matrices(matrices) {}

    template <typename OutStream>
    void to_stream_simple(OutStream& outfile, const std::string& quiverName = "Q") const {
        index num_vert = degrees.size();
        index num_edges = edges.size();     
        // Define the quiver
        
        outfile << quiverName << " := Quiver(" << num_vert << ", [ ";
        // Write edges
        for (index j = 0; j < num_edges; j++) {
            index source = edges[j].first;
            index target = edges[j].second;
            // Write edge
            outfile << "["  << source + 1 << "," << target + 1 << ", \"E" << j << "\"" << "], ";
        }

        outfile.seekp(-2, std::ios::end);  // Remove trailing comma
        outfile << "]);" << std::endl;

        // Define module
        // syntax: M, [1,2,2, ..], [ .. ["b", [[1,0], [1,1]] ], ..] )
        outfile << quiverName << ", [";
        // dimension vector
        for (index i = 0; i < num_vert; i++) {
            outfile << dimensionVector[i] << ",";
        }
        outfile.seekp(-1, std::ios::end);  // Remove trailing comma
        outfile << "], [ ";
        // matrices at edges
        bool remove_comma = false;
        for (index j = 0; j < num_edges; j++) {
            auto source = edges[j].first;
            auto target = edges[j].second;
            
            const SparseMatrix<index>* current_mat_ptr = &matrices[j];
            // We write our matrices column wise. If you want the row format, uncomment the next lines.
            // SparseMatrix transposed = matrices[j].transposed_copy();
            // current_mat_ptr = &transposed;
            const auto& current_mat = *current_mat_ptr;

            // Check dimensions again
            if (current_mat.get_num_cols() != dimensionVector[source]) {
                throw std::runtime_error("Dimension mismatch for source vertex " + std::to_string(source));
            }
            if (current_mat.get_num_rows() != dimensionVector[target]) {
                throw std::runtime_error("Dimension mismatch for target vertex " + std::to_string(target));
            }

            if(dimensionVector[source] == 0 || dimensionVector[target] == 0){
                continue;
            } else {
                remove_comma = true;
            }
            // Write matrix at arrow; where each vector is a row

            
            outfile << "[\"E" << j << "\", [";
            for (index i = 0; i < current_mat.get_num_cols(); i++) {
                outfile << "[";
                auto it = current_mat.data[i].begin();

                for (index k = 0; k < current_mat.get_num_rows(); k++) {
                    if (it != current_mat.data[i].end()) {
                        if(*it == k){
                            outfile << "1" << ",";
                            ++it;
                        } else {
                            outfile << "0" << ",";
                        }
                    } else {
                        outfile << "0" << ",";
                    }
                }
                outfile.seekp(-1, std::ios::end);  // Remove trailing comma
                outfile << "],";
            }
            outfile.seekp(-1, std::ios::end);  // Remove trailing comma
            outfile << "] ], ";
        }
        if(remove_comma){
            outfile.seekp(-2, std::ios::end);  // Remove trailing comma
        }
        outfile << " ];" << std::endl;
    }

    template <typename OutStream>
    void to_streamQPA(OutStream& outfile, const std::string& quiverName = "Q") {
        
        index dimension = 0;
        for(index i = 0; i < dimensionVector.size(); i++){
            dimension += dimensionVector[i];
        }

        index num_vert = degrees.size();
        index num_edges = edges.size();

        // Write header information -  May need to be changed
        // outfile << "LoadPackage(\"qpa\");\n\n";

        // Define the quiver
        outfile << "# Dimension of Module: " << dimension << "\n";
        outfile << quiverName << " := Quiver(" << num_vert << ", [ ";
        

        // Write arrows (edges)
        for (index j = 0; j < num_edges; j++) {
            index source = edges[j].first;
            index target = edges[j].second;

            // Write arrow
            outfile << "["  << source + 1 << "," << target + 1 << ", \"E" << j << "\"" << "], ";

            
            }

        outfile.seekp(-2, std::ios::end);  // Remove trailing comma
        outfile << "]);" << std::endl;

        // Define path algebra
        outfile << "A" << quiverName << " := PathAlgebra(GF(2), " << quiverName<< ");" << std::endl;

        // Define module
        // syntax: N := RightModuleOverPathAlgebra( A, [1,2,2], [ .. ["b", [[1,0], [-1,0]] ], ..] )
        // Use Z(2)^0 and 0*Z(2) for 1 and 0; (Need Z(p^k) for general finite field)

        outfile << "M" << quiverName << " := RightModuleOverPathAlgebra(A" << quiverName << ", [";
        for (index i = 0; i < num_vert; i++) {
            outfile << dimensionVector[i] << ",";
        }
        outfile.seekp(-1, std::ios::end);  // Remove trailing comma
        outfile << "], [ ";

        for (index j = 0; j < num_edges; j++) {
            auto source = edges[j].first;
            auto target = edges[j].second;
            // No need for this, qpa wants the transposed matrices in row format, so we just write our matrices column wise instead of row wise
            // No need for: SparseMatrix edgeMatrix = matrices[j].transposed_copy();

            // Check dimensions again
            if (matrices[j].get_num_cols() != dimensionVector[source]) {
                throw std::runtime_error("Dimension mismatch for source vertex " + std::to_string(source));
            }
            if (matrices[j].get_num_rows() != dimensionVector[target]) {
                throw std::runtime_error("Dimension mismatch for target vertex " + std::to_string(target));
            }

            if(dimensionVector[source] == 0 || dimensionVector[target] == 0){
                continue;
            }
            // Write matrix at arrow; where each vector is a row

            
            outfile << "[\"E" << j << "\", [";
            for (index i = 0; i < matrices[j].get_num_cols(); i++) {
                outfile << "[";
                auto it = matrices[j].data[i].begin();

                for (index k = 0; k < matrices[j].get_num_rows(); k++) {
                    if (it != matrices[j].data[i].end()) {
                        if(*it == k){
                            outfile << "Z(2)^0" << ",";
                            ++it;
                        } else {
                            outfile << "0*Z(2)" << ",";
                        }
                    } else {
                        outfile << "0*Z(2)" << ",";
                    }
                }
                outfile.seekp(-1, std::ios::end);  // Remove trailing comma
                outfile << "],";
            }
            outfile.seekp(-1, std::ios::end);  // Remove trailing comma
            outfile << "] ], ";
        }

        outfile.seekp(-2, std::ios::end);  // Remove trailing comma
        outfile << " ]);" << std::endl;
    }

    void write_QPA_decomp(const std::string& fullFilePath, const std::string& quiverName = "Q") {
        
        std::string scriptPath = fullFilePath + ".g";
        std::string logFilePath = fullFilePath + "_log.txt";

        std::ofstream outfile(scriptPath);
        if (!outfile.is_open()) {
            std::cerr << "Error opening file at path: " << scriptPath << std::endl;

        }
        outfile << "LoadPackage(\"qpa\");\n";
        outfile << "start_read := Runtime();  # Record the starting time\n";

        to_streamQPA(outfile, quiverName);
        
        outfile << "List" << quiverName << " := BlockDecompositionOfModule(M" << quiverName << ");" << std::endl;
        outfile << "end_read := Runtime(); \n";
        outfile << "time_taken := end_read - start_read; \n";
        outfile << "PrintTo( \"" << logFilePath << " \" , \"CPU time taken to read file and compute decomposition: \", time_taken, \" milliseconds\", List" << quiverName << "); \n";
        outfile << "QUIT;";
        std::cout << "Quiver file written to: " << fullFilePath << std::endl;

    }
};

} // namespace graded_linalg