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

#include <function_delaunay/Miniball.hpp>

#include <CGAL/Min_sphere_of_spheres_d_traits_d.h>
#include <CGAL/Min_sphere_of_spheres_d.h>


namespace function_delaunay {

  struct Meb_accessor_Miniball {

    Meb_accessor_Miniball() {}
    
    typedef std::vector<std::vector<double>>::const_iterator Point_iterator; 
    typedef std::vector<double>::const_iterator Coord_iterator;
    
    typedef Miniball::
    Miniball <Miniball::CoordAccessor<Point_iterator, Coord_iterator> > 
      Miniball;
    
    // I see no good way of doing this without a pointer
    Miniball* meb;

    Point_iterator last_begin,last_end;
    
    void new_meb(Point_iterator begin, Point_iterator end) {
      int d = begin->size();
      last_begin=begin;
      last_end=end;
      meb=new Miniball(d,begin,end);

    }

    void get_indices_of_support(std::vector<int> &indices) {
      for(auto support_it = meb->support_points_begin();support_it!=meb->support_points_end();support_it++) {
	int count = 0;
	for(Point_iterator list_it=last_begin;list_it!=last_end;list_it++) {
	  if(*support_it==list_it) {
	    indices.push_back(count);
	    break;
	  }
	  count++;
	}
      }
      std::sort(indices.begin(),indices.end());
    }

    double get_radius() {
      return sqrt(meb->squared_radius());
    }
    
  };



  template<typename Kernel_,int d>
    struct Meb_accessor_CGAL {
            
      typedef Kernel_ Kernel;
      typedef CGAL::Min_sphere_of_spheres_d_traits_d<Kernel,double,d> Min_sphere_traits;
      typedef CGAL::Min_sphere_of_spheres_d<Min_sphere_traits> Miniball;
      typedef typename Kernel::Construct_point_d Construct_point;
      //typedef typename Kernel::Point Point;
      typedef typename Min_sphere_traits::Sphere Sphere;
      
      Miniball meb;
      std::vector<Sphere> spheres;

      template<typename InputIterator>
      void new_meb(InputIterator begin, InputIterator end) {
		
	spheres.clear();
	
	for(auto it = begin;it!=end;it++) {
	  spheres.push_back(std::make_pair(Construct_point()(it->begin(),it->end()),0));
	}
	meb.set(spheres.begin(),spheres.end());
	
      }
	//std::cout << "Number of support points: " << std::distance(meb.support_begin(),meb.support_end()) << std::endl;
	
	/*
	  for(auto it = meb.support_begin();it!=meb.support_end();it++) {
	  for(int i =0;i<DIM;i++) {
	  std::cout << ((*it).first)[i] << " ";
	  }
	  std::cout << std::endl;
	  }
	*/

    void get_indices_of_support(std::vector<int> &indices) {
      for(auto support_it = meb.support_begin();support_it!=meb.support_end();support_it++) {
	int count=0;
	for(auto list_it=spheres.begin();list_it!=spheres.end();list_it++) {
	  if(*support_it==*list_it) {
	    indices.push_back(count);
	    break;
	  }
	  count++;
	}
      }
      std::sort(indices.begin(),indices.end());
    }
	
      double get_radius() {
	return CGAL::to_double(meb.radius());
      }
      
    };

#if 0
  template<typename Kernel_2>
    struct Compute_meb_radius_with_CGAL_2 {
      
      
      typedef Kernel_2 Kernel;
      typedef CGAL::Min_sphere_of_spheres_d_traits_2<Kernel,double> Min_sphere_traits;
      typedef CGAL::Min_sphere_of_spheres_d<Min_sphere_traits> Min_sphere;
      typedef typename Kernel::Point_2 Point;
      
      template<typename InputIterator>
      double operator()(InputIterator begin, InputIterator end) {
	
	typedef typename Min_sphere_traits::Sphere Sphere;
	
	std::vector<Sphere> spheres;
	
	for(auto it = begin;it!=end;it++) {
	  spheres.push_back(std::make_pair(Point((*it)[0],(*it)[1]),0));
	}
	Min_sphere meb(spheres.begin(),spheres.end());
	
	//std::cout << "Number of support points: " << std::distance(meb.support_begin(),meb.support_end()) << std::endl;
	
	/*
	  for(auto it = meb.support_begin();it!=meb.support_end();it++) {
	  for(int i =0;i<DIM;i++) {
	  std::cout << ((*it).first)[i] << " ";
	  }
	  std::cout << std::endl;
	  }
	*/
	
	return CGAL::to_double(meb.radius());
      }
      
    };



#endif

#if 0
  
  // Brute-force method, very slow in its current version - not used (needs to be brought into accessor format
  template<typename Kernel_>
    struct Compute_meb_radius_with_brute_force {
      
      typedef Kernel_ Kernel;
      typedef typename Kernel::Point_d Point;
      typedef typename Kernel::FT FT;
      typedef typename Kernel::Side_of_bounded_sphere_d Side_of_bounded_sphere;
      typedef typename Kernel::Contained_in_simplex_d Contained_in_simplex;
      typedef typename Kernel::Squared_distance_d Square_distance;
      typedef typename Kernel::Compute_squared_radius_d Compute_squared_radius;
      typedef typename Kernel::Construct_circumcenter_d Construct_circumcenter;
      
      std::pair<int,int> diametral_pair() {
	FT curr_max(0);
	std::pair<int,int> max_pair;
	for(int i=0;i<n;i++) {
	  for(int j=i+1;j<n;j++) {
	    FT new_dist = sq_dist(_data[i],_data[j]);
	    //std::cout << "Sq dist for " << i << " " << j << ": " << CGAL::to_double(new_dist) << std::endl;
	    if(new_dist > curr_max) {
	      curr_max=new_dist;
	      max_pair=std::make_pair(i,j);
	    }	  
	  }
	}
	return max_pair;
      }
      
      // This function is perhaps problematic, as the construction of the circumcenter is inexact
      template<typename InputIterator>
      bool contains_circumcenter(InputIterator begin,InputIterator end) {
	
    //std::cout << "Circum: " << CGAL::to_double(circumcenter(begin,end)[0]) << " " << CGAL::to_double(circumcenter(begin,end)[1]) << " " << std::endl;
	
	return is_in_simplex(begin,end,circumcenter(begin,end));
      }
      

      template<typename InputIterator>
      bool is_enclosing(InputIterator begin,InputIterator end) {
	for(auto pit = _data.begin();pit!=_data.end();pit++) {
	  if(side_of_sphere(begin,end,*pit)==CGAL::ON_UNBOUNDED_SIDE) {
	    return false;
	  }
	}
	return true;
      }
      
      bool check(std::vector<Point>& points) {
	/*
	  for(Point p : points) {
	  for(int i=0;i<dim;i++) {
	  std::cout << p[i] << " ";
	  }
	  std::cout << std::endl;
	  }
	  
	  if(! contains_circumcenter(points.begin(),points.end()) ) {
	  std::cout << "Does not contain center" << std::endl;
	  }
	  if(! is_enclosing(points.begin(),points.end())) {
	  std::cout << "not enclosing" << std::endl;
	  }
	*/
	return contains_circumcenter(points.begin(),points.end()) && is_enclosing(points.begin(),points.end());
      }
      
      int update_at(std::vector<int>& indices,int pos,std::vector<Point>& points) {
	if(pos==-1) {
	  std::cout << "SOMETHING IS WRONG, I SHOULD NOT BE HERE" << std::endl;
	  return -1;
	}
	indices[pos]++;
	for(int i=pos;i<=dim;i++) {
	  indices[i]=indices[pos]+i-pos;
	  if(indices[i]>=n) {
	    return update_at(indices,pos-1,points);
	  }
	}
	return pos;
      }
      
      bool update(std::vector<int>& indices, std::vector<Point>& points) {
	int pos = update_at(indices,dim,points);
	if(pos==-1) {
	  return false;
	}
	//std::cout << "update_at returned " << pos << std::endl;
	if(points.size()<=dim-pos) {
	  //std::cout << "Resize point vector to " << dim-pos+1 << std::endl;
	  points.resize(dim-pos+1);
	}
	for(int i=pos;i<=dim;i++) {
	  //std::cout << "i=" << i << " indices[i]=" << indices[i] << std::endl;
	  points[dim-i]=_data[indices[i]];
	}
	{
	  // All assertion code
	  int count=0;
	  for(int i=0;i<indices.size();i++) {
	    if(indices[i]!=-1) {
	      count++;
	    }
	  }
	  CGAL_assertion(count==points.size());
	  for(int i=0;i<indices.size();i++) {
	    if(indices[i]!=-1) {
	      CGAL_assertion(points[dim-i]==_data[indices[i]]);
	    }
	  }
	}
	return true;
      }
      
      
      template<typename InputIterator>
      double operator() (InputIterator begin, InputIterator end) {
	_data.clear();
	for(InputIterator it=begin;it!=end;it++) {
	  _data.push_back(Point(it->begin(),it->end()));
	}
	this->n = _data.size();
	CGAL_assertion(n>=1);
	this->dim = typename Kernel::Point_dimension_d()(_data[0]);
	
	if(n==1) {
	  return 0.0;
	}
	std::pair<int,int> diam_pair = diametral_pair();
	//std::cout << "Diametral pair is " << diam_pair.first << " " << diam_pair.second << std::endl;
	
	std::vector<Point> vec_diam;
	vec_diam.push_back(_data[diam_pair.first]);
	vec_diam.push_back(_data[diam_pair.second]);
	if(is_enclosing(vec_diam.begin(),vec_diam.end())) {
	  return sqrt(CGAL::to_double(sq_dist(_data[diam_pair.first],_data[diam_pair.second])))/2.;
	}
	CGAL_assertion(n>=3);
	std::vector<int> curr_indices;
	std::vector<Point> curr_points;
	for(int i=0;i<dim-2;i++) {
	  curr_indices.push_back(-1);
	}
	curr_indices.push_back(0);
	curr_points.push_back(_data[2]);
	curr_indices.push_back(1);
	curr_points.push_back(_data[1]);
	curr_indices.push_back(2);
	curr_points.push_back(_data[0]);
	
	double result;
	while(true) {
	  /*
	    std::cout << "At pos ";
	    for(int i=0;i<curr_indices.size();i++) {
	    std::cout << curr_indices[i] << " ";
	    }
	    std::cout << std::endl;
	  */
	  if(check(curr_points)) {
	    return sqrt(CGAL::to_double(sq_radius(curr_points.begin(),curr_points.end())));
	  } else {
	    if(! update(curr_indices,curr_points)) {
	      break;
	    }
	  }
	}
	return 0.0;
      }
      
      protected:
      
      int dim;
      int n;
      std::vector<Point> _data;
      Side_of_bounded_sphere side_of_sphere;
      Contained_in_simplex is_in_simplex;
      Square_distance sq_dist;
      Compute_squared_radius sq_radius;
      Construct_circumcenter circumcenter;
      
    };
#endif

}
