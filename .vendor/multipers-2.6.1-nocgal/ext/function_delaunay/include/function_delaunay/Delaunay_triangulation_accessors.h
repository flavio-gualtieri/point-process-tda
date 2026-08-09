/* Copyright 2023 TU Graz
   Author: Michael Kerber
   
   This file is part of function_delaunay
   
   function_delaunay is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   function_delaunay is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   
   You should have received a copy of the GNU General Public License
   along with function_delaunay.  If not, see <https://www.gnu.org/licenses/>.
*/


#pragma once

//#define CGAL_EIGEN3_ENABLED
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>

#include <CGAL/Epick_d.h>
#include <CGAL/Epeck_d.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Delaunay_triangulation.h>

#include <function_delaunay/Point_with_densities.h>

namespace function_delaunay {

  struct Vertex_info {
    //double density;
    int idx;
  };

  template<typename DimTag>
    struct Delaunay_triangulation_accessor_d {
      
      typedef DimTag Dim_tag;

      typedef CGAL::Epick_d< Dim_tag >  Kernel;
      //typedef CGAL::Epeck_d< Dim_tag >  Kernel;

      typedef CGAL::Triangulation_data_structure<Dim_tag, CGAL::Triangulation_vertex<Kernel,Vertex_info>,CGAL::Triangulation_full_cell<Kernel> > TDS;
      typedef CGAL::Delaunay_triangulation<Kernel,TDS> Triangulation;

      typedef typename Triangulation::Vertex_handle Vertex_handle;
      typedef typename Triangulation::Full_cell_handle Full_cell_handle;
      typedef typename Triangulation::Point Point;
    

      Triangulation T;

      int d;

      Delaunay_triangulation_accessor_d(int d) : T(d), d(d) {

	Vertex_handle inf = T.infinite_vertex();
	inf->data().idx=-1;
	//inf->data().density=-999.999;
      }

      Point get_point(Point_with_densities& p) {
	return Point(p.x.begin(),p.x.end());
      }
      
      int current_dimension() {
	return T.current_dimension();
      }

      bool is_full_dimensional() {
	return current_dimension()==d;
      }

      Full_cell_handle locate(Point& p) {
	return T.locate(p);
      }

      void compute_conflict_zone(Point p,Full_cell_handle loc,std::vector<Full_cell_handle>& cells) {
	T.compute_conflict_zone(p,loc,std::back_inserter(cells));
      }

      bool is_infinite(Vertex_handle vh) {
	return T.is_infinite(vh);
      }

      bool is_infinite(Full_cell_handle c) {
	return T.is_infinite(c);
      }
      
      Vertex_handle insert(Point& p) {
	  //std::cout << "DEL inserting " << p.cartesian(0) << ", " << p.cartesian(1) << std::endl;
	  return T.insert(p);
      }

      void remove(Vertex_handle vh) {
	  //std::cout << "DEL Removing" << vh->point().cartesian(0) << ", " << vh->point().cartesian(1) << std::endl;
	  T.remove(vh);
      }
      
      Vertex_handle insert(Point& p, Full_cell_handle c) {
	  //std::cout << "DEL inserting with cell" << p.cartesian(0) << ", " << p.cartesian(1) << std::endl;
	return T.insert(p,c);
      }

      template<typename InputIterator>
      void insert(InputIterator begin, InputIterator end) {
	  //std::cout << "DEL inserting range" << std::endl;
	T.insert(begin,end);
      }

        
      typename Triangulation::Finite_vertex_iterator vertices_begin() {
	return T.finite_vertices_begin();
      }

      typename Triangulation::Finite_vertex_iterator vertices_end() {
	return T.finite_vertices_end();
      }

      typename Triangulation::Finite_full_cell_iterator finite_full_cells_begin() {
	return T.finite_full_cells_begin();
      }

      typename Triangulation::Finite_full_cell_iterator finite_full_cells_end() {
	return T.finite_full_cells_end();
      }

      // This must be templated because some Vertex-iterators cannot be casted to Vertex_handle
      template<typename VertexHandle>
      Vertex_info& data_of_vertex(VertexHandle vh) {
	return vh->data();
      }
      
        template<typename VertexHandle>
        void get_adjacent_vertex_ids(VertexHandle vh, std::vector<int>& ids)
        {
            if (current_dimension() == 0)
                return;

            // std::cout << "Vertex handle with data " << data_of_vertex(vh).idx << '\n';

            std::vector<typename Triangulation::Full_cell_handle> cells;
            T.incident_full_cells(vh, std::back_inserter(cells));

            std::unordered_set<int> seen;

            for (auto cell : cells) {
                if (T.is_infinite(cell))
                    continue;

                int d = T.current_dimension();;
                for (int i = 0; i < d + 1; ++i) {
                    auto vh2 = cell->vertex(i);
                    if (vh2 == vh || T.is_infinite(vh2))
                        continue;

                    int idx = data_of_vertex(vh2).idx;
                    if (seen.insert(idx).second)
                        ids.push_back(idx);
                }
            }
        }
    };

    struct Delaunay_triangulation_accessor_2 {
      
      typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;

      //typedef CGAL::Exact_predicates_exact_constructions_kernel Kernel;
      typedef CGAL::Triangulation_vertex_base_with_info_2<Vertex_info,Kernel> Vb;
      typedef CGAL::Triangulation_face_base_2<Kernel> Fb;
      typedef CGAL::Triangulation_data_structure_2<Vb,Fb> Tds;
      typedef CGAL::Delaunay_triangulation_2<Kernel,Tds> Triangulation;

      typedef typename Triangulation::Vertex_handle Vertex_handle;
      typedef typename Triangulation::Face_handle Full_cell_handle;
      typedef typename Triangulation::Point Point;
    

      Triangulation T;

      Delaunay_triangulation_accessor_2() {

	Vertex_handle inf = T.infinite_vertex();
	inf->info().idx=-1;
	//inf->info().density=-999.999;
      }

      Point get_point(Point_with_densities& p) {
	return Point(p.x[0],p.x[1]);
      }
      
      int current_dimension() {
	return T.dimension();
      }

      bool is_full_dimensional() {
	return current_dimension()==2;
      }
      

      Full_cell_handle locate(Point& p) {
	return T.locate(p);
      }

      void compute_conflict_zone(Point p,Full_cell_handle loc,std::vector<Full_cell_handle>& cells) {
	T.get_conflicts(p,std::back_inserter(cells),loc);
      }

      bool is_infinite(Vertex_handle vh) {
	return T.is_infinite(vh);
      }

      bool is_infinite(Full_cell_handle c) {
	return T.is_infinite(c);
      }
      
      Vertex_handle insert(Point& p) {
	return T.insert(p);
      }

      Vertex_handle insert(Point& p, Full_cell_handle c) {
	return T.insert(p,c);
      }

      template<typename InputIterator>
      void insert(InputIterator begin, InputIterator end) {
	T.insert(begin,end);
      }

      void remove(Vertex_handle vh) {
	  T.remove(vh);
      }
	
      typename Triangulation::Finite_vertices_iterator vertices_begin() {
	return T.finite_vertices_begin();
      }

      typename Triangulation::Finite_vertices_iterator vertices_end() {
	return T.finite_vertices_end();
      }

      typename Triangulation::Finite_faces_iterator finite_full_cells_begin() {
	return T.finite_faces_begin();
      }

      typename Triangulation::Finite_faces_iterator finite_full_cells_end() {
	return T.finite_faces_end();
      }

      Vertex_info& data_of_vertex(Vertex_handle vh) {
	return vh->info();
      }


      template<typename VertexHandle>
      void get_adjacent_vertex_ids(VertexHandle vh,std::vector<int>& ids) {
	  if(current_dimension()==0) {
	      return;
	  }
	  // std::cout << "Vertex handle with data " << data_of_vertex(vh).idx << std::endl;
	  typename Triangulation::Vertex_circulator circ_start=T.incident_vertices(vh),circ=circ_start;
	  if(!is_infinite(circ)) {
	      ids.push_back(data_of_vertex(circ).idx);
	  }
	  circ++;
	  while(circ!=circ_start) {
	      if(!is_infinite(circ)) {
		  ids.push_back(data_of_vertex(circ).idx);
	      }
	      circ++;
	  }
      }
	  

  
    };

    struct Delaunay_triangulation_accessor_3 {
      
      typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
      //typedef CGAL::Exact_predicates_exact_constructions_kernel Kernel;
      typedef CGAL::Triangulation_vertex_base_with_info_3<Vertex_info,Kernel> Vb;
      typedef CGAL::Triangulation_cell_base_3<Kernel>               Fb;
      typedef CGAL::Triangulation_data_structure_3<Vb,Fb>      Tds;
      typedef CGAL::Delaunay_triangulation_3<Kernel,Tds,CGAL::Fast_location> Triangulation;

      typedef typename Triangulation::Vertex_handle Vertex_handle;
      typedef typename Triangulation::Cell_handle Full_cell_handle;
      typedef typename Triangulation::Point Point;
      typedef typename Triangulation::Facet Facet;
      typedef typename Triangulation::Edge Edge;

      Triangulation T;

      Delaunay_triangulation_accessor_3() {

	Vertex_handle inf = T.infinite_vertex();
	inf->info().idx=-1;
	//inf->info().density=-999.999;
      }

      Point get_point(Point_with_densities& p) {
	return Point(p.x[0],p.x[1],p.x[2]);
      }
      
      int current_dimension() {
	return T.dimension();
      }

      bool is_full_dimensional() {
	return current_dimension()==3;
      }
      

      Full_cell_handle locate(Point& p) {
	return T.locate(p);
      }

      void compute_conflict_zone(Point p,Full_cell_handle loc,std::vector<Full_cell_handle>& cells) {
	std::vector<Facet> facets;
	T.find_conflicts(p,loc,std::back_inserter(facets),std::back_inserter(cells));
      }

      bool is_infinite(Vertex_handle vh) {
	return T.is_infinite(vh);
      }

      bool is_infinite(Full_cell_handle c) {
	return T.is_infinite(c);
      }

      
      Vertex_handle insert(Point& p) {
	return T.insert(p);
      }

      Vertex_handle insert(Point& p, Full_cell_handle c) {
	return T.insert(p,c);
      }

      template<typename InputIterator>
      void insert(InputIterator begin, InputIterator end) {
	T.insert(begin,end);
      }

      void remove(Vertex_handle vh) {
	  T.remove(vh);
      }
	
      typename Triangulation::Finite_vertices_iterator vertices_begin() {
	return T.finite_vertices_begin();
      }

      typename Triangulation::Finite_vertices_iterator vertices_end() {
	return T.finite_vertices_end();
      }

      typename Triangulation::Finite_cells_iterator finite_full_cells_begin() {
	return T.finite_cells_begin();
      }

      typename Triangulation::Finite_cells_iterator finite_full_cells_end() {
	return T.finite_cells_end();
      }
  
      Vertex_info& data_of_vertex(Vertex_handle vh) {
	return vh->info();
      }



        template <typename VertexHandle>
        void get_adjacent_vertex_ids(VertexHandle vh, std::vector<int>& ids)
        {
            if (current_dimension() == 0)
                return;

            std::vector<VertexHandle> neighbors;
            T.finite_adjacent_vertices(vh, std::back_inserter(neighbors));

            ids.reserve(neighbors.size());
            for (const auto& nv : neighbors) {
                    ids.push_back(nv->info().idx);
            }

            std::sort(ids.begin(), ids.end());
            ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        }
    };

}
