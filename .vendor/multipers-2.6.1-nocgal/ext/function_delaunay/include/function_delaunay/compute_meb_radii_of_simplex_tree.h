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

#include <function_delaunay/Meb_accessors.h>
#include <CGAL/Epick_d.h>

#define COMPUTE_MEB_DIMENSION_DISPATCH(d)				\
    typedef CGAL::Epick_d<CGAL::Dimension_tag<d> > Kernel;		\
    typedef Meb_accessor_CGAL<Kernel,d> Meb_functor;			\
    Meb_functor meb_accesor;						\
    _compute_meb_radii_of_simplex_tree<SimplexTree,Meb_functor>(simplex_tree,meb_accesor,input_points);

namespace function_delaunay {

    bool contains(std::vector<int>& v, int x) {
	auto it= std::lower_bound(v.begin(),v.end(),x);
	return it!=v.end() && *it==x;
    }

    template<typename SimplexTree,typename SimplexKey> void _assign_filtration_values_in_interval(SimplexTree &simplex_tree,
												  SimplexKey upper,
												  std::vector<int> &lower,
												  double meb_radius) {
	/*
	  std::cout << "Start recursion: \n";
	  auto vertices = simplex_tree.simplex_vertex_range(upper);
	  std::cout << "Upper: ";
	  for(auto vit=vertices.begin();vit!=vertices.end();vit++) {
	  std::cout << *vit << " ";
	  }
	  std::cout << std::endl;
	  std::cout << "Lower: ";
	  for(auto vit=lower.begin();vit!=lower.end();vit++) {
	  std::cout << *vit << " ";
	  }
	  std::cout << std::endl;
	*/
	auto bd = simplex_tree.boundary_opposite_vertex_simplex_range(upper);
	for(auto bd_it=bd.begin();bd_it!=bd.end();bd_it++) {
	    auto curr_pair = *bd_it;
	    auto bd_simp = curr_pair.first;
	    if(simplex_tree.filtration(bd_simp)!=-1) {
		continue;
	    }
	    int vert=(int)curr_pair.second;
	    if(!contains(lower,vert)) { // still in interval
		//typename SimplexTree::Boundary_simplex_range bd_simp = curr_pair.first;
	
		auto vertices = simplex_tree.simplex_vertex_range(bd_simp);
		/*
		  std::cout << "Assigningfilt val to: ";
		  for(auto vit=vertices.begin();vit!=vertices.end();vit++) {
		  std::cout << *vit << " ";
		  }
		  std::cout << std::endl;
		*/
		simplex_tree.assign_filtration(bd_simp,meb_radius);
		_assign_filtration_values_in_interval(simplex_tree,bd_simp,lower,meb_radius);
	    }
	}
    
	/*
	  auto vertices = simplex_tree.simplex_vertex_range(curr_pair.first);
	  std::cout << "Pair first: ";
	  for(auto vit=vertices.begin();vit!=vertices.end();vit++) {
	  std::cout << *vit << " ";
	  }
	  std::cout << "\nPair second: " << curr_pair.second << std::endl;
	*/
    
    }

    template<typename SimplexTree,typename MebAccessor> void _compute_meb_radii_of_simplex_tree(SimplexTree &simplex_tree,
												MebAccessor &meb_accessor,
												const std::vector<Point_with_densities> &input_points) {


#if FUNCTION_DELAUNAY_SMART_MEB_TRAVERSAL

	auto all_simplices = simplex_tree.complex_simplex_range();

    
	// Needed to ensure that all filtration values are unset
	for(auto it=all_simplices.begin();it!=all_simplices.end();it++) {
	    simplex_tree.assign_filtration(*it,-1);
	}

	long number_of_meb_computations=0;
	int simplex_tree_dim = simplex_tree.dimension();
	for(int k=simplex_tree_dim;k>=0;k--) {
	    auto st_skeleton_range=simplex_tree.skeleton_simplex_range(k);
	    for(typename SimplexTree::Skeleton_simplex_iterator it = st_skeleton_range.begin();it!=st_skeleton_range.end();it++) {
		if(simplex_tree.filtration(*it)==-1 && simplex_tree.dimension(*it)==k) {
		    auto vertices = simplex_tree.simplex_vertex_range(*it);
		    std::vector<std::vector<double>> points;
      
		    //std::cout << "Simplex has dimension " << simplex_tree.dimension(*it) << std::endl;  
		    //std::cout << "Simplex has key " << simplex_tree.key(*it) << std::endl;
		    for(auto vit=vertices.begin();vit!=vertices.end();vit++) {
			//std::cout << *vit << " ";
			points.push_back(input_points[*vit].x);
		    }
		    meb_accessor.new_meb(points.begin(), points.end());
		    number_of_meb_computations++;

		    std::vector<int> support;
		    meb_accessor.get_indices_of_support(support);
		    assert(support.size()<=points.size());
		    std::vector<int> lower;
		    int count=0;
		    auto support_it=support.begin();
		    for(auto vit=vertices.begin();vit!=vertices.end();vit++) {
			if(support_it==support.end()) {
			    break;
			}
			if(count==*support_it) {
			    support_it++;
			    lower.push_back(*vit);
			}
			count++;
		    }
		    assert(support.size()==lower.size());
		    auto lower_key = simplex_tree.find(lower);
		    assert(lower_key!=SimplexTree::null_simplex());
		    double meb_radius=simplex_tree.filtration(lower_key);
		    if(meb_radius==-1) {
			meb_radius = meb_accessor.get_radius();
		    } 
		    /*
		      else {
		      std::cout << "Radius already known, not taking the computed radius" << std::endl;
		      }
		    */
		    simplex_tree.assign_filtration(*it,meb_radius);
		    //std::cout << "Computed meb of " << points.size() << " points and received " << support.size() << " support points" << std::endl;
		    std::sort(lower.begin(),lower.end());	
		    if(points.size()>lower.size()) {
			_assign_filtration_values_in_interval<SimplexTree>(simplex_tree,*it,lower,meb_radius);
		    }
		}
	    }
	}
	std::cout << "Computed " << number_of_meb_computations << " mebs (number of simplices=" << simplex_tree.num_simplices() << ")" << std::endl;

#else // the naive "go over all simplices and compute meb solution
	auto all_simplices = simplex_tree.complex_simplex_range();
    
	for(auto it=all_simplices.begin();it!=all_simplices.end();it++) {
      
	    auto vertices = simplex_tree.simplex_vertex_range(*it);
      
	    std::vector<std::vector<double>> points;
      
	    //std::cout << "Simplex has dimension " << simplex_tree.dimension(*it) << std::endl;  
	    //std::cout << "Simplex has key " << simplex_tree.key(*it) << std::endl;
	    for(auto vit=vertices.begin();vit!=vertices.end();vit++) {
		//std::cout << *vit << " ";
		points.push_back(input_points[*vit].x);
	    }
	    //std::cout << std::endl;
	    meb_accessor.new_meb(points.begin(), points.end());
	    double meb_radius = meb_accessor.get_radius();
	    //std::cout << "Meb is " << meb_radius << std::endl;
	    simplex_tree.assign_filtration(*it,meb_radius);
	}
#endif
#if !NDEBUG
	auto former_precision = std::cout.precision();
	std::cout.precision(std::numeric_limits<double>::max_digits10);

	for(auto it=all_simplices.begin();it!=all_simplices.end();it++) {
      
	    if(simplex_tree.dimension(*it)>0) {
	
		double filt_value = simplex_tree.filtration(*it);

		for(auto bd_it=simplex_tree.boundary_simplex_range(*it).begin();bd_it!=simplex_tree.boundary_simplex_range(*it).end();bd_it++) {
		    double filt_bd = simplex_tree.filtration(*bd_it);
		    bool bad_order = filt_value<filt_bd;
		    bool very_close_unequal = filt_value>filt_bd && (filt_value-filt_bd<0.000000000001);
		    if(bad_order) {
			std::cout << "Found bad pair" << std::endl;
		    }
		    if(very_close_unequal) {
			std::cout << "Found very close pair" << std::endl;
		    }
		    if(bad_order || very_close_unequal) {
	    
			auto vertices = simplex_tree.simplex_vertex_range(*it);
			std::cout << "Simplex: ";
			for(auto vit=vertices.begin();vit!=vertices.end();vit++) {
			    std::cout << *vit << " ";
			}
			std::cout << ", value: " << filt_value << std::endl;
			std::cout << "Face: ";
			vertices = simplex_tree.simplex_vertex_range(*bd_it);
			for(auto vit=vertices.begin();vit!=vertices.end();vit++) {
			    std::cout << *vit << " ";
			}
			std::cout << ", value: " << filt_bd << std::endl;
		    }
		}
	    }
	}
	std::cout.precision(former_precision);
#endif

    }

    // Just for debugging
    template<typename SimplexTree,typename Meb_accessor1,typename Meb_accessor2> void _compare_meb_radii_of_simplex_tree(SimplexTree &simplex_tree,
															 Meb_accessor1 &meb1,
															 Meb_accessor2 &meb2,
															 const std::vector<Point_with_densities> &input_points) {
	auto all_simplices = simplex_tree.complex_simplex_range();
    
	for(auto it=all_simplices.begin();it!=all_simplices.end();it++) {
      
	    auto vertices = simplex_tree.simplex_vertex_range(*it);
      
	    std::vector<std::vector<double>> points;
      
	    //std::cout << "Simplex has dimension " << simplex_tree.dimension(*it) << std::endl;  
	    //std::cout << "Simplex has key " << simplex_tree.key(*it) << std::endl;
	    for(auto vit=vertices.begin();vit!=vertices.end();vit++) {
		//std::cout << *vit << " ";
		points.push_back(input_points[*vit].x);
	    }
	    //std::cout << std::endl;
	    meb1.new_meb(points.begin(), points.end());
	    double meb_radius1 = meb1.get_radius();
	    meb2.new_meb(points.begin(), points.end());
	    double meb_radius2 = meb2.get_radius();
	    if(std::abs(meb_radius1-meb_radius2)>0.0000001) {
		std::cout << "Discrepancy: " << std::abs(meb_radius1-meb_radius2) << std::endl;
	    }
	    simplex_tree.assign_filtration(*it,meb_radius1);
	}
    }
    

    template<typename SimplexTree> void compute_meb_radii_of_simplex_tree(SimplexTree &simplex_tree,
									  const std::vector<Point_with_densities> &input_points) {
	if(input_points.size()==0) {
	    return;
	}
	int d = input_points[0].dimension();

#if !FUNCTION_DELAUNAY_ALWAYS_USE_MINIBALL   
	// CGAL is more efficient, but requires the dimension in the class fixed.
	// For all dimensions up to 10, we dispatch like this
	// Note that for d=2 and d=3, there is a specialized traits class for Min_sphere_d
	// but the tests did not show any difference in performance
	if(d==2) {
	    COMPUTE_MEB_DIMENSION_DISPATCH(2)
		} else if(d==3) {
	    COMPUTE_MEB_DIMENSION_DISPATCH(3)
		} else if(d==4) {
	    COMPUTE_MEB_DIMENSION_DISPATCH(4)
		/* Commented out to save compile time
		   } else if(d==5) {
		   COMPUTE_MEB_DIMENSION_DISPATCH(5)
		   } else if(d==6) {
		   COMPUTE_MEB_DIMENSION_DISPATCH(6)
		   } else if(d==7) {
		   COMPUTE_MEB_DIMENSION_DISPATCH(7)
		   } else if(d==8) {
		   COMPUTE_MEB_DIMENSION_DISPATCH(8)
		   } else if(d==9) {
		   COMPUTE_MEB_DIMENSION_DISPATCH(9)
		   } else if(d==10) {
		   COMPUTE_MEB_DIMENSION_DISPATCH(10)
		*/
		} else {
#else
	    {
#endif
		// If you really want this, use Miniball
		typedef Meb_accessor_Miniball Meb_accessor;
		Meb_accessor meb_accessor;
		_compute_meb_radii_of_simplex_tree<SimplexTree,Meb_accessor>(simplex_tree,meb_accessor,input_points);
	    }
	}
    

    }
