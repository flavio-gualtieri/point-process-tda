/* Copyright 2025 TU Graz
   Author: Michael Kerber
   
   This file is part of multi_critical
   
   multi_critical is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   multi_critical is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.
   
   You should have received a copy of the GNU Lesser General Public License
   along with multi_critical.  If not, see <https://www.gnu.org/licenses/>.*/

#pragma once

#include<algorithm>
#include <unordered_map>
#include <unordered_set>

#include<mpp_utils/create_graded_matrices_from_pre_column_struct.h>
#include<mpp_utils/sorting_utility.h>

#include<multi_critical/basic.h>
#include<multi_critical/Grade.h>


namespace multi_critical {



    // Represents a linear map
    typedef std::unordered_map<index,std::vector<index>> Map;

    // The first index is the dimension
    typedef std::vector<Map> Dim_Map;

    // Computes g o f (x)
#if 0
    void compose_maps(Map& g, Map& f, index x, std::vector<index>& result) {
	//std::cout << "Compose starts"  << std::endl;
	std::vector<index>& fx = f[x];
	std::unordered_set<index> container;
	for(index y : fx) {
	    std::vector<index>& gfx = g[y];
	    for(index z : gfx) {
		if(!container.count(z)) {
		    container.insert(z);
		} else {
		    container.erase(z);
		}
	    }
	    //std::cout << "Now container has size " << container.size() << std::endl;
	}
	std::copy(container.begin(),container.end(),std::back_inserter(result));
	std::sort(result.begin(),result.end());
    }

#else
   void compose_maps(Map& g, Map& f, index x, std::vector<index>& result) {
	//std::cout << "Compose starts"  << std::endl;
	std::vector<index>& fx = f[x];
	std::vector<index> container;
	for(index y : fx) {
	    std::vector<index>& gfx = g[y];
	    std::copy(gfx.begin(),gfx.end(),std::back_inserter(container));
	}
	std::sort(container.begin(),container.end());
	// Remove duplicates
	index container_it=0;
	while(container_it<container.size()) {
	    if(container_it<container.size()-1 && container[container_it]==container[container_it+1]) {
		container_it+=2;
		continue;
	    }
	    result.push_back(container[container_it]);
	    container_it++;
	}

    } 
#endif
    
    void sum_of_composed_maps(Map& g1, Map& f1, Map& g2, Map& f2, index x, std::vector<index>& result) {
	std::vector<index> part1,part2;
	compose_maps(g1,f1,x,part1);
	compose_maps(g2,f2,x,part2);
	index it1=0,it2=0;
	index n1=part1.size();
	index n2=part2.size();
	while(it1<n1 || it2<n2) {
	    if(it1==n1) {
		result.push_back(part2[it2]);
		it2++;
		continue;
	    }
	    if(it2==n2) {
		result.push_back(part1[it1]);
		it1++;
		continue;
	    }
	    if(part1[it1]==part2[it2]) {
		it1++;
		it2++;
		continue;
	    }
	    if(part1[it1]<part2[it2]) {
		result.push_back(part1[it1]);
		it1++;
		continue;
	    }
	    if(part1[it1]>part2[it2]) {
		result.push_back(part2[it2]);
		it2++;
		continue;
	    }
				       
	}
    }
	    
	    
    // This object is needed to sort relations in the syzygy creation routine
    template<typename InputComplex, typename OutputComplex>
    struct Relation_sorter {

	InputComplex& input_complex;
	OutputComplex& output_complex;
	Dim_Map& p1;
	
	index dim;
	
	Relation_sorter(InputComplex& input_complex,
			OutputComplex& output_complex,
			Dim_Map& p1)
	    : input_complex(input_complex),output_complex(output_complex), p1(p1) {
	}

	// Needed to set to pick the right dimension of relations in the comparison
	void set_dim(index dim) {
	    this->dim=dim;
	}

	/* We don't need that part of the code because the relations are sorted
	   by index in their width
	
	index priority(index rel_idx) {
	    std::vector<index>& p1_image = p1[dim][rel_idx];
	    //std::cout << "Size of " << rel_idx << " is : " << p1_image.size() << std::endl;
	    assert(p1_image.size()==2);
	    assert(output_complex[dim-1][p1_image[0]].copy_of == output_complex[dim-1][p1_image[1]].copy_of);
	    index first_copy_idx = output_complex[dim-1][p1_image[0]].copy_idx;
	    index second_copy_idx = output_complex[dim-1][p1_image[1]].copy_idx;
	    assert(second_copy_idx > first_copy_idx);
	    return second_copy_idx - first_copy_idx;
	}
	
	bool operator() (index rel_idx1, index rel_idx2) {
	    index prio1 = priority(rel_idx1);
	    index prio2 = priority(rel_idx2);
	    if(prio1<prio2) {
		return true;
	    }
	    if(prio1>prio2) {
		return false;
	    }
	    return rel_idx1<rel_idx2;
	}
	*/

	void find_preimage_of_syzygies(std::vector<index>& image_of_map,
				       std::vector<index>& result) {
	    if(image_of_map.size()==0) {
		//std::cout << "Composition is empty" << std::endl;
		return;
	    }
	    //std::cout << "Comp is not empty, size= " << image_of_map.size() << ", do something" << std::endl;
	    
	    int image_it = 0;
	    index idx_of_curr_input_simp = output_complex[dim][image_of_map[image_it]].copy_of;
	    std::vector<index> rels_of_input_simp;
	    while(image_it<image_of_map.size()) {
		//std::cout << "Here, Size of " << image_of_map[image_it] << " is " << p1[dim][image_of_map[image_it]].size() << std::endl;
		assert(p1[dim][image_of_map[image_it]].size()==2);
		if(output_complex[dim][image_of_map[image_it]].copy_of==idx_of_curr_input_simp) {
		    rels_of_input_simp.push_back(image_of_map[image_it]);
		} else {
		    /*
		    {
			std::cout << "Collected " << rels_of_input_simp.size() << " relations to fill in " << std::endl;
			std::cout << "Instance: ";
			for(auto x : rels_of_input_simp) {
			    std::cout << x << " ";
			}
			std::cout << std::endl;
		    }
		    */
		    input_complex[dim-1][idx_of_curr_input_simp].get_syzygies_of_cycle(rels_of_input_simp,result);
		    //std::cout << "Syzygies found" << std::endl;
		    rels_of_input_simp.clear();
		    rels_of_input_simp.push_back(image_of_map[image_it]);
		    idx_of_curr_input_simp=output_complex[dim][image_of_map[image_it]].copy_of;
		}
		image_it++;
		
	    }
	    assert(!rels_of_input_simp.empty());
	    //std::cout << "Collected, at the end now, remaining " <<rels_of_input_simp.size() << " relations"   << std::endl;
	    input_complex[dim-1][idx_of_curr_input_simp].get_syzygies_of_cycle(rels_of_input_simp,result);
	    std::sort(result.begin(),result.end());
	    //std::cout << "Image consists of " << result.size() << " triangles" << std::endl;
	}
    };

    
    struct Input_simplex {
	std::vector<Grade> grades;
	std::vector<index> boundary;
	std::vector<index> copies;
	// relations[i] is the list of pairs (j,idx) with i<j and idx a relation joining i and j
	std::vector<std::vector<std::pair<index,index>>> relations;
	// path mode only creates adjacent relations, so direct lookup avoids repeated scans
	bool path_uses_only_adjacent_relations=false;
	std::vector<index> adjacent_relations;
	// Syzygy with relation idx as upper boundary is saved with key idx
	std::unordered_map<index,index> syzygies;
	// The two other relations bounding the syzygy are saved here
	std::unordered_map<index,std::pair<index,index>> lower_hull_of_syzygy;
	
	Input_simplex() {}
	Input_simplex(std::vector<Grade>& grades,
		      std::vector<long>& boundary) {
	    std::copy(grades.begin(),grades.end(),std::back_inserter(this->grades));
	    std::copy(boundary.begin(),boundary.end(),std::back_inserter(this->boundary));
	    
	}
	void get_path_of_relations(int i,int j, std::vector<index>& rels) {
	    assert(i<=j);
	    if(i==j) {
		return;
	    }
	    if(path_uses_only_adjacent_relations) {
		assert(j<=adjacent_relations.size());
		for(int k=i;k<j;k++) {
		    assert(adjacent_relations[k]>=0);
		    rels.push_back(adjacent_relations[k]);
		}
		return;
	    }
	    int next_copy=0;
	    int next_rel=0;
	    for(auto pair : relations[i]) {
		if(pair.first>j) {
		    break;
		}
		next_copy=pair.first;
		next_rel=pair.second;
	    }
	    rels.push_back(next_rel);
	    get_path_of_relations(next_copy,j,rels);
	}
	
	index get_relation_of_copies(int i, int j) {
	    assert(i>=0 && j>=0);
	    assert(i<copies.size());
	    assert(j<copies.size());
	    for(auto p : relations[i]) {
		if(p.first==j) {
		    return p.second;
		}
	    }
	    return -1;
	}

	void get_syzygies_of_cycle(std::vector<index>& rels, std::vector<index>& result) {
	    // Little optimization: If there are 3 edges, it must be a "simple" cycle,
	    // and we can avoid the priority queue
	    if(rels.size()==3) {
		assert(rels[0]<rels[1] && rels[1]<rels[2]);
		result.push_back(syzygies[rels[2]]);
		return;
	    }
	    
	    /*
	    std::cout << "Set up pq with " << rels.size() << " elements: ";
	    for(index i : rels) {
		std::cout << i << " ";
	    }
	    std::cout << std::endl;
	    */
	    std::priority_queue<index> queue(rels.begin(),rels.end());
	    //td::cout << "Done" << std::endl;
	    //int pushed_elements=0;
	    while(!queue.empty()) {
		index next=queue.top();
		queue.pop();
		if(queue.top()==next) {
		    //there was a duplicate, remove it and go on
		    queue.pop();
		    continue;
		}
		/*
		std::cout << "next=" << next << std::endl;
		  {
		
		  
		  std::cout << "Number of copies: " << copies.size() << std::endl;
		  
		  std::cout << "Relations: " << std::endl;
		  for(auto x : relations) {
		  for(auto y : x) {
		  std::cout << y.second << " ";
		  }
		  }
		  std::cout << std::endl;
		  
		  std::cout << "Syzygy Container contains: " << std::endl;
		  for(auto p: syzygies) {
		  std::cout << p.first << "-> " << p.second << std::endl;
		  }
		*/
	    

		result.push_back(syzygies[next]);
		//pushed_elements++;
		//std::cout << "Lower hull Container contains: " << std::endl;
		/*
		for(auto p : lower_hull_of_syzygy) {
		    std::cout << p.first << "-> " << p.second.first << " " << p.second.second << std::endl;
		}
		*/
		std::pair<index,index>& lower_hull = lower_hull_of_syzygy[syzygies[next]];
		//std::cout << "Pushing " << lower_hull.first << " and " << lower_hull.second << std::endl;
		queue.push(lower_hull.first);
		queue.push(lower_hull.second);
	    }
	    //std::cout << "Resulted in " << pushed_elements << " pushed triangles" << std::endl;
	}
	
    };

    void debug_print_input_complex(std::vector<std::vector<Input_simplex>>& complex) {
	for(int i=0;i<complex.size();i++) {
	    std::cout << "Dimension " << i << std::endl;
	    for(int j=0;j<complex[i].size();j++) {
		std::cout << "Dimension/Index: " << i << " " << j << std::endl; 
		Input_simplex& s = complex[i][j];
		std::cout << "Grades: " << std::endl;
		for(Grade& gr : s.grades) {
		    std::cout << gr.x << " " << gr.y << std::endl;
		}
		std::cout << "Boundary: " << std::flush;
		for(index idx : s.boundary) {
		    std::cout << idx << " ";
		}
		std::cout << std::endl;
		std::cout << "Copies: " << std::flush;
		for(index idx : s.copies) {
		    std::cout << idx << " ";
		}
		std::cout << std::endl;
	    }
	}
    }

    enum Type {
	GENERATOR,
	RELATION,
	SYZYGY
    };
    
    
    struct Output_simplex {
	Grade grade;
	std::vector<index> boundary;
	index copy_of;
	index copy_idx;
	// Can be a copy of a generator or a relation joining two copies
	Type type;
	Output_simplex() {}
	Output_simplex(Grade& g, std::vector<index> bd, index copy_idx) {
	    grade=g;
	    std::copy(bd.begin(),bd.end(),std::back_inserter(boundary));
	    copy_of=copy_idx;
	}
    };

    
  template<typename Matrix_Grade>
  class Output_complex_accessor {

  public:
      
      typedef Matrix_Grade Grade;

      typedef std::vector<std::vector<Output_simplex>> Output_complex;
      
      Output_complex& output_complex;

      std::vector<std::vector<index>>& permutations;
      std::vector<std::vector<index>>& inv_permutations;
      
      Output_complex_accessor(Output_complex& output_complex,
			      std::vector<std::vector<index>>& permutations,
			      std::vector<std::vector<index>>& inv_permutations)
	  : output_complex(output_complex), permutations(permutations), inv_permutations(inv_permutations) {
	  /*
	  std::cout << "Output complex size: " << output_complex.size() << std::endl;
	  std::cout << "Output complex sizes: ";for(auto x : output_complex) std::cout << x.size() << " ";std::cout << std::endl;
	  std::cout << "Perm size: " << permutations.size() << std::endl;
	  std::cout << "Inv Perm size: " << inv_permutations.size() << std::endl;
	  std::cout << "Perm sizes: ";for(auto x : permutations) std::cout << x.size() << " ";std::cout << std::endl;
	  std::cout << "Inv Perm sizes: ";for(auto x : inv_permutations) std::cout << x.size() << " ";std::cout << std::endl;
	  */
      }

      int number_of_parameters() {
	  return 2;
      }
      
      int number_of_matrices() {
	  return output_complex.size();
      }

      // We have to reverse the order in the output complex (also in all what follows)
      int number_of_columns(int i) {
	  return output_complex[number_of_matrices()-i-1].size();
      }

      Grade get_grade(int i, int j) {
	  //std::cout << "get grade " << i << " " << j << std::endl;
	  Grade g(output_complex[number_of_matrices()-i-1][permutations[number_of_matrices()-i-1][j]].grade.x,
		  output_complex[number_of_matrices()-i-1][permutations[number_of_matrices()-i-1][j]].grade.y);
	  //std::cout << "Done get_grade" << std::endl;
	  return g;
      }

      void get_boundary(int i, int j,std::vector<index>& result) {
	  //std::cout << "get boundary" << std::endl;
	  // We have to translate the boundary according to the inverse permutation
	  if(i==number_of_matrices()-1) {
	      return;
	  }
	  std::vector<index>& bd=output_complex[number_of_matrices()-i-1][permutations[number_of_matrices()-i-1][j]].boundary;
	  result.resize(bd.size());
	  for(int k=0;k<bd.size();k++) {
	      result[k]=inv_permutations[number_of_matrices()-i-2][bd[k]];
	  }
	  std::sort(result.begin(),result.end());
      }

      void clear(int i) {
	  //pre_columns[number_of_matrices()-i-1].clear();
	  //pre_columns[number_of_matrices()-i-1].shrink_to_fit();
      }
      
  };
  
    
    void debug_print_output_complex(std::vector<std::vector<Output_simplex>>& complex) {
	for(int i=0;i<complex.size();i++) {
	    std::cout << "Dimension " << i << std::endl;
	    for(int j=0;j<complex[i].size();j++) {
		std::cout << "Dimension/Index: " << i << " " << j << std::endl; 
		Output_simplex& s = complex[i][j];
		std::cout << "Grade: " << s.grade.x << " " << s.grade.y << std::endl;
		std::cout << "Boundary: " << std::flush;
		for(index idx : s.boundary) {
		    std::cout << idx << " ";
		}
		std::cout << std::endl;
		std::cout << "Copy of: " << s.copy_of << std::endl;
		std::cout << "Type: ";
		if(s.type==GENERATOR) {
		    std::cout << "Generator" << std::endl;
		}
		if(s.type==RELATION) {
		    std::cout << "Relation" << std::endl;
		}
		if(s.type==SYZYGY) {
		    std::cout << "Syzygy" << std::endl;
		}
	    }
	}
    }

    struct Grade_sorter {

	std::vector<Output_simplex>& data;

	Grade_sorter(std::vector<Output_simplex>& data) : data(data) {}

	bool operator() (int i, int j) {
	    Grade& g1 = data[i].grade;
	    Grade& g2 = data[j].grade;
	    if(g1.x<g2.x) {
		return true;
	    }
	    if(g1.x>g2.x) {
		return false;
	    }
	    if(g1.y<g2.y) {
		return true;
	    }
	    if(g1.y>g2.y) {
		return false;
	    }
	    return i<j;
	}
    };


    
    void boundary_of_boundaries(std::vector<std::vector<Output_simplex>>& complex, int ell, int idx, std::vector<index>& bd_of_bd) {
	
	Output_simplex& output_simplex=complex[ell][idx];
	for(index bd_idx : output_simplex.boundary) {
	    Output_simplex& bd_output_simplex=complex[ell-1][bd_idx];
	    std::copy(bd_output_simplex.boundary.begin(),
		      bd_output_simplex.boundary.end(),
		      std::back_inserter(bd_of_bd));
	}
	std::sort(bd_of_bd.begin(),bd_of_bd.end());
    }
	
	
    void check_output_complex(std::vector<std::vector<Output_simplex>>& complex) {

	// Check that boundary of vertices is empty
	for(int idx=0;idx<complex[0].size();idx++) {
	    if(complex[0][idx].boundary.size()!=0) {
		std::cerr << "Boundary in dim 0 is not empty!!" << std::endl;
	    }
	}
	
	// Check whether \partial\partial=0
	for(int ell=2;ell<complex.size();ell++) {
	    for(int idx=0;idx<complex[ell].size();idx++) {
		Output_simplex& output_simplex=complex[ell][idx];
		std::vector<index> bd_of_bd;
		boundary_of_boundaries(complex,ell,idx,bd_of_bd);
		/*
		{
		    std::cout << "BD_OF_BD: ";
		    for(index idx : bd_of_bd) {
			std::cout << idx << " ";
		    }
		    std::cout << std::endl;
		}
		*/
		if(bd_of_bd.size()%2!=0) {
		    std::cerr << "Bd of bd has odd number!" << std::endl;
		}
		for(int j=0;j<bd_of_bd.size();j+=2) {
		    if(bd_of_bd[j]!=bd_of_bd[j+1]) {
			std::cerr << "Bd of bd is not zero for dim=" << ell << ", idx=" << idx << std::endl;
			std::cerr << "Type of simplex: " << output_simplex.type << std::endl;
			std::cerr << "Content: " ;
			for(index p : bd_of_bd) {
			    std::cout << p << " ";
			}
			std::cout << std::endl;
		    }
		}
	    }
	}

	// Check grade consistency
	for(int ell=1;ell<complex.size();ell++) {
	    for(int idx=0;idx<complex[ell].size();idx++) {
		Output_simplex& output_simplex=complex[ell][idx];
		Grade& gr = output_simplex.grade;
		for(index bd_idx : output_simplex.boundary) {
		    Output_simplex& bd_output_simplex=complex[ell-1][bd_idx];
		    Grade bd_gr = bd_output_simplex.grade;
		    if(!( bd_gr <= gr)) {
			std::cerr << "Simplex (" << ell << ", " << idx << ") has grade " << gr.x << " " << gr.y << " but has grade " << bd_gr.x << " " << bd_gr.y << " in its boundary" << std::endl;
		    }
		}
	    }
	}
	
    }


    	    
	
    
    
    template<typename ParserType, typename GradedMatrix>
	void free_resolution(ParserType& parser,
			     std::vector<GradedMatrix>& result,
			     bool use_logpath=true) {


	
	  
	
	
	int d = parser.number_of_parameters();
	
	if(d!=2) {
	    std::cerr << "This code expects a 2-parameter input" << std::endl;
	    return;
	}
	
	int L = parser.number_of_levels();

	if(multi_critical::verbose) std::cout << "Number of levels: " << L << std::endl;

	std::vector<index> idx_of_level_input,idx_of_level_output;
	for(int i=0;i<=L+1;i++) {
	    idx_of_level_input.push_back(0);
	    idx_of_level_output.push_back(0);
	}
	

	// input_complex[i][j] = j-th simplex in level i
	typedef std::vector<std::vector<Input_simplex>> Input_complex;
	
	Input_complex input_complex;
	input_complex.resize(L);

	typedef std::vector<std::vector<Output_simplex>> Output_complex;
	
	Output_complex output_complex;
	output_complex.resize(L+2);

	// These are presentation maps. p1: Rels->Gens, p2: Syz->Rels
	// With the path approach, p2 is not needed
	Dim_Map p1, p2;
	p1.resize(L+1);
	p2.resize(L+2);
	
	// These are the boundary maps, f0: Gens->Gens, f1: Rels->Rels, f2: Syz->Syz
	Dim_Map f0, f1, f2;
	f0.resize(L);
	f1.resize(L+1);
	f2.resize(L+2);
	
	// The 1st level homotopy maps. h0:Gens->Rels, h1: Rels->Syz
	Dim_Map h0,h1;
	h0.resize(L);
	h1.resize(L+1);
	
	// The 2nd level homotopy map hh: Gens->Syz
	Dim_Map hh;
	hh.resize(L);


	

#if MULTI_CRITICAL_TIMERS
	 multi_critical::prepare_input_complex_timer.start();
#endif

	if(multi_critical::verbose) std::cout << "Prepare input_complex" << std::endl;
	for(int ell=L-1;ell>=0;ell--) {
	    if(multi_critical::very_verbose) std::cout << "Dimension " << ell << std::endl;
	    while(parser.has_next_column(ell+1)) {
		std::vector<double> raw_grades;
		std::vector<std::pair<long,int>> boundary_and_coeffs;
		parser.next_column(ell+1,
				   std::back_inserter(raw_grades),
				   std::back_inserter(boundary_and_coeffs));
		// Ignore coefficients
		std::vector<long> boundary;
		for(auto& x : boundary_and_coeffs) {
		    boundary.push_back(x.first);
		}
		std::vector<Grade> grades;
		get_incomparable_grades_in_lex_order(raw_grades,grades);
		Input_simplex new_input_simplex(grades, boundary);
		index idx_of_new_input_simplex=idx_of_level_input[ell];

		input_complex[L-1-ell].push_back(new_input_simplex);
		idx_of_level_input[ell]++;
	    }
	}


#if MULTI_CRITICAL_TIMERS
	multi_critical::prepare_input_complex_timer.stop();
#endif

	
	//debug_print_input_complex(input_complex);

	if(multi_critical::verbose) std::cout << "Build 1-critical Generators and presentation maps"
					      << std::endl;

#if MULTI_CRITICAL_TIMERS
	multi_critical::prepare_output_complex_timer.start();
#endif
	 
	for(int ell=0;ell<L;ell++) {
	    if(multi_critical::very_verbose) std::cout << "Dimension " << ell << std::endl;
	    for(index idx_of_input_simplex=0;idx_of_input_simplex<input_complex[ell].size();idx_of_input_simplex++) {

		Input_simplex& input_simplex =input_complex[ell][idx_of_input_simplex];
		std::vector<Grade>& grades = input_simplex.grades;
		//std::cout << "New input simplex with " << grades.size() << " copies" << std::endl;		
		//std::cout << "Generator is " << grades.size() << "-critical" << std::endl;
		// Generate the basis elements
		for(int j=0;j<grades.size();j++) {
		    Grade& gr=grades[j];
		    index idx_of_new_output_simplex=idx_of_level_output[ell];
		    Output_simplex new_output_simplex;
		    new_output_simplex.type=GENERATOR;
		    new_output_simplex.grade=gr;
		    new_output_simplex.copy_of=idx_of_input_simplex;
		    new_output_simplex.copy_idx=j;
		    // Boundary is determined later
		    assert(idx_of_new_output_simplex>=0);
		    input_simplex.copies.push_back(idx_of_new_output_simplex);
		    output_complex[ell].push_back(new_output_simplex);
		    idx_of_level_output[ell]++;
		}
		// Generate relations and syzygies
		int width=1;
		input_simplex.path_uses_only_adjacent_relations=!use_logpath;
		input_simplex.relations.resize(input_simplex.copies.size());
		if(input_simplex.path_uses_only_adjacent_relations && input_simplex.copies.size()>1) {
		    input_simplex.adjacent_relations.assign(input_simplex.copies.size()-1,-1);
		}
		// It is important for later that the relations are generated in increasing
		// length, that is, the width of the relations (for a single input simplex)
		// increase with their index
		while(width < grades.size()) {
		    for(int j=width;j<grades.size();j+=width) {
			//std::cout << "Creating relation " << j-width << " " << j << std::endl;
			Output_simplex new_output_simplex_for_relation;
			index idx_of_new_output_relation=idx_of_level_output[ell+1];
			idx_of_level_output[ell+1]++;
			new_output_simplex_for_relation.type=RELATION;
			new_output_simplex_for_relation.grade=join(grades[j-width],grades[j]);
			new_output_simplex_for_relation.copy_of=idx_of_input_simplex;
			assert(input_simplex.copies[j-width] < input_simplex.copies[j]);
			//std::cout << "HERE " << input_simplex.copies[j-width] << " " << input_simplex.copies[j] << std::endl;
			std::vector<index> image;
			image.push_back(input_simplex.copies[j-width]);
			image.push_back(input_simplex.copies[j]);
			p1[ell+1][idx_of_new_output_relation]=image;
						
			output_complex[ell+1].push_back(new_output_simplex_for_relation);
			input_simplex.relations[j-width].push_back(std::make_pair(j,idx_of_new_output_relation));
			if(input_simplex.path_uses_only_adjacent_relations) {
			    assert(width==1);
			    input_simplex.adjacent_relations[j-width]=idx_of_new_output_relation;
			}
			
			
			if(width>1) {
			    Output_simplex new_output_simplex_for_syzygy;
			    index idx_of_new_output_syzygy=idx_of_level_output[ell+2];
			    idx_of_level_output[ell+2]++;
			    new_output_simplex_for_syzygy.type=SYZYGY;
			    new_output_simplex_for_syzygy.grade=new_output_simplex_for_relation.grade;
			    new_output_simplex_for_syzygy.copy_of=idx_of_input_simplex;
			    index lower_rel_idx_1 = input_simplex.get_relation_of_copies(j-width,j-width/2);
			    index lower_rel_idx_2 = input_simplex.get_relation_of_copies(j-width/2,j);
			    p2[ell+2][idx_of_new_output_syzygy].push_back(lower_rel_idx_1);
			    p2[ell+2][idx_of_new_output_syzygy].push_back(lower_rel_idx_2);
			    p2[ell+2][idx_of_new_output_syzygy].push_back(idx_of_new_output_relation);
			    assert(p2[ell+2][idx_of_new_output_syzygy][0]<p2[ell+2][idx_of_new_output_syzygy][1]);
			    assert(p2[ell+2][idx_of_new_output_syzygy][1]<p2[ell+2][idx_of_new_output_syzygy][2]);
			    output_complex[ell+2].push_back(new_output_simplex_for_syzygy);
			    input_simplex.syzygies[idx_of_new_output_relation]=idx_of_new_output_syzygy;
			    input_simplex.lower_hull_of_syzygy[idx_of_new_output_syzygy]=std::make_pair(lower_rel_idx_1,lower_rel_idx_2);
			}
		    }
		    if(!use_logpath) {
			// Only use path, so width>1 is not allowed
			break;
		    }
		    width*=2;
		}
	    }
	}
	

#if MULTI_CRITICAL_TIMERS
	multi_critical::prepare_output_complex_timer.stop();
#endif
	

	
	//debug_print_input_complex(input_complex);
	    
	
	// Determine the rest of the boundaries

	// First choose a consistent map between the generators
	if(multi_critical::verbose) std::cout << "Constructing map f0" << std::endl;

#if MULTI_CRITICAL_TIMERS
	multi_critical::compute_f0_timer.start();
#endif
	
	
	for(int ell=1;ell<L;ell++) {
	    if(multi_critical::very_verbose) std::cout << "Dimension " << ell << std::endl;
	    for(Input_simplex& input_simplex : input_complex[ell]) {
		std::vector<index>& bd = input_simplex.boundary;
		/*
		{
		    std::cout << "Here I am at " << ell << ", " << std::endl;
		    
		    
		    std::cout << "Boundary (" << bd.size() << "): ";
		    for(index x : bd ) {
			std::cout << x << " " << std::flush;
		    }
		    std::cout << std::endl;
		}
		*/
		for(int i=0; i< input_simplex.copies.size(); i++) {
		    index output_simplex_idx = input_simplex.copies[i];
		    Output_simplex& output_simplex = output_complex[ell][output_simplex_idx];
		    assert(output_simplex.type==GENERATOR);
		    Grade& gr_of_copy = output_simplex.grade;
		    std::vector<index> image;
		    for(index idx : bd) {
			//std::cout << "bd index: " << idx << std::endl;
			Input_simplex& bd_input_simplex=input_complex[ell-1][idx];
			assert(bd_input_simplex.grades.size()>=1);

			index idx_copy =get_copy_smaller_equal_than(bd_input_simplex.grades,gr_of_copy);
			assert(bd_input_simplex.copies[idx_copy]>=0);
			image.push_back(bd_input_simplex.copies[idx_copy]);
			//std::cout << "Adding to bd on level " << ell << std::endl;
		    }
		    f0[ell][output_simplex_idx]=image;
		}
	    }
	}

#if MULTI_CRITICAL_TIMERS
	multi_critical::compute_f0_timer.stop();
#endif


	
	//debug_print_output_complex(output_complex);
	
	
	// Now compute the map between relations
	if(multi_critical::verbose) std::cout << "Constructing map f1" << std::endl;
	
#if MULTI_CRITICAL_TIMERS
	multi_critical::compute_f1_timer.start();
#endif

	for(int ell=L-1;ell>=1;ell--) {
	    long count_inst=0;		
	    if(multi_critical::very_verbose) std::cout << "Dim ell=" << ell << std::endl;
	    for(Input_simplex& input_simplex : input_complex[ell]) {
		std::vector<index>& bd = input_simplex.boundary;
		for(int i=0; i<input_simplex.copies.size(); i++) {
		    for(auto p : input_simplex.relations[i]) {
			count_inst++;
			int j = p.first;
			assert(i<j);
			index relation_idx = p.second;
			Output_simplex& output_relation = output_complex[ell+1][relation_idx];
		    
			assert(output_relation.type==RELATION);
			// Get the two generators that create the relation

			index output_gen_smaller_idx = input_simplex.copies[i];
			index output_gen_larger_idx = input_simplex.copies[j];
			Output_simplex& output_gen_smaller = output_complex[ell][output_gen_smaller_idx];
			Output_simplex& output_gen_larger = output_complex[ell][output_gen_larger_idx];

			std::vector<index>& p1_of_rel = p1[ell+1][relation_idx];
			assert(p1_of_rel.size()==2);
			assert(p1_of_rel[0]==output_gen_smaller_idx);
			assert(p1_of_rel[1]==output_gen_larger_idx);
			    
			assert(output_gen_smaller.type==GENERATOR);
			assert(output_gen_larger.type==GENERATOR);
			std::vector<index> combined_bd;
			std::copy(f0[ell][output_gen_smaller_idx].begin(),f0[ell][output_gen_smaller_idx].end(),std::back_inserter(combined_bd));
			std::copy(f0[ell][output_gen_larger_idx].begin(),f0[ell][output_gen_larger_idx].end(),std::back_inserter(combined_bd));
			std::sort(combined_bd.begin(),combined_bd.end());

			/*
			{
			    std::cout << "Info (" << ell << "): " << std::flush;
			    
			    for(index idx : combined_bd) {
				std::cout << idx << " ";
			    }
			    std::cout << std::endl;
			}
			*/

			std::vector<index> image;
			for(int j=0;j<combined_bd.size();j+=2) {
			    Output_simplex& first_copy = output_complex[ell-1][combined_bd[j]];
			    Output_simplex& second_copy = output_complex[ell-1][combined_bd[j+1]];
			    //std::cout << "Copies " << combined_bd[j] << " " << combined_bd[j+1] << std::endl;
			    //std::cout << "Copy of " << first_copy.copy_of << " " << second_copy.copy_of << std::endl;
			    assert(first_copy.copy_of==second_copy.copy_of);
			    assert(first_copy.copy_idx<=second_copy.copy_idx);
			    assert(first_copy.type==GENERATOR);
			    assert(second_copy.type==GENERATOR);
			    Input_simplex& input_simplex = input_complex[ell-1][first_copy.copy_of];
			    assert(input_simplex.copies[first_copy.copy_idx]==combined_bd[j]);
			    assert(input_simplex.copies[second_copy.copy_idx]==combined_bd[j+1]);

			    std::vector<index> local_image;
			    input_simplex.get_path_of_relations(first_copy.copy_idx, second_copy.copy_idx, local_image);
			    /*
			    {
				std::cout << "Relation path: ";
				for(index rel_idx : local_image) {
				    std::cout << "( " << output_complex[ell-1][p1[ell][rel_idx][0]].copy_idx << " " << output_complex[ell-1][p1[ell][rel_idx][1]].copy_idx << " )" << std::flush;
				}
				std::cout << std::endl;
			    }
			    */
			    std::copy(local_image.begin(),local_image.end(),std::back_inserter(image));
			}
			std::sort(image.begin(),image.end());
			f1[ell+1][relation_idx]=image;
		    }
		}
	    }
	    if(multi_critical::very_verbose) std::cout << "Handled " << count_inst << " instances" << std::endl;
	}


#if MULTI_CRITICAL_TIMERS
	multi_critical::compute_f1_timer.stop();
#endif

	if(multi_critical::verbose) std::cout << "Constructing map h0 " << std::endl;
#if MULTI_CRITICAL_TIMERS
	multi_critical::compute_h0_timer.start();
#endif
	// And finally the map from generators to relations in codim 2
	for(int ell=L-1;ell>=2;ell--) {
	    long count_inst=0;
	    if(multi_critical::very_verbose) std::cout << "At dimension " << ell << std::endl;
	    for(Input_simplex& input_simplex : input_complex[ell]) {
		/*
		{
		    std::cout << "AT: ";
		    for(index idx : input_simplex.boundary) {
			std::cout << idx << " ";
		    }
		    std::cout << std::endl;
		}
		*/
		for(int i=0; i<input_simplex.copies.size(); i++) {
		    count_inst++;
		    index copy_idx = input_simplex.copies[i];
		    std::vector<index> h_circ_h_of_copy_idx;
		    compose_maps(f0[ell-1],f0[ell],copy_idx,h_circ_h_of_copy_idx);
		    /*
		    std::cout << "Size of hh_copy_idx: " << h_circ_h_of_copy_idx.size() << std::endl;
		    
		    {
			std::cout << "Info2 (" << ell << "): " << std::flush;
			
			for(index idx : h_circ_h_of_copy_idx) {
			    std::cout << idx << " ";
			}
			std::cout << std::endl;
		    }
		    */
		    std::vector<index> image;
		    for(int j=0;j<h_circ_h_of_copy_idx.size();j+=2) {
			Output_simplex& first_copy = output_complex[ell-2][h_circ_h_of_copy_idx[j]];
			Output_simplex& second_copy = output_complex[ell-2][h_circ_h_of_copy_idx[j+1]];
			assert(first_copy.type==GENERATOR);
			assert(second_copy.type==GENERATOR);
			assert(first_copy.copy_of==second_copy.copy_of);
			assert(first_copy.copy_idx<=second_copy.copy_idx);
			Input_simplex& input_simplex_of_bd = input_complex[ell-2][first_copy.copy_of];
			assert(input_simplex_of_bd.copies[first_copy.copy_idx]==h_circ_h_of_copy_idx[j]);
			assert(input_simplex_of_bd.copies[second_copy.copy_idx]==h_circ_h_of_copy_idx[j+1]);
			std::vector<index> local_image;
			input_simplex_of_bd.get_path_of_relations(first_copy.copy_idx, second_copy.copy_idx, local_image);
			std::copy(local_image.begin(),local_image.end(), std::back_inserter(image));
			
		    }
		    std::sort(image.begin(),image.end());
		    h0[ell][copy_idx]=image;
	    	}
	    }
	    if(multi_critical::very_verbose) std::cout << "Handled " << count_inst << " instances" << std::endl;
	}
#if MULTI_CRITICAL_TIMERS
	multi_critical::compute_h0_timer.stop();
#endif

	Relation_sorter<Input_complex, Output_complex> relation_sorter(input_complex,output_complex,p1);


	
#if MULTI_CRITICAL_TIMERS
	multi_critical::compute_f2_timer.start();
#endif
	if(use_logpath) {
	    
	    if(multi_critical::verbose) std::cout << "Now compute f2" << std::endl;
	    
	    for(int ell=L+1;ell>=3;ell--) {
		if(multi_critical::very_verbose) std::cout << "Dim=" << ell << std::endl;
		
		long no_elements_with_non_trivial_preimage=0;
		
		relation_sorter.set_dim(ell-2);
		
		// Iterating over all syzygies is very inconvenient and indirect.
		// TODO: Store the output complex separately in the fields
		// generators, relations, syzygies, each indexed by dimension,
		// and only merge in the end
		for(Input_simplex& input_simplex : input_complex[ell-2]) {
		    for(auto& p : input_simplex.syzygies) {
			index idx_of_syz = p.second;
			std::vector<index> p_circ_f;
			compose_maps(f1[ell-1],p2[ell],idx_of_syz,p_circ_f);
			std::vector<index> preimage;
			relation_sorter.find_preimage_of_syzygies(p_circ_f,preimage);
			if(!preimage.empty()) {
			    f2[ell][idx_of_syz]=preimage;
			    no_elements_with_non_trivial_preimage++;
			}
		    }
		    
		}
		if(multi_critical::very_verbose) std::cout << no_elements_with_non_trivial_preimage << " elements have non-trivial function value" << std::endl;
	    }
	}

#if MULTI_CRITICAL_TIMERS
	multi_critical::compute_f2_timer.stop();
#endif

#if MULTI_CRITICAL_TIMERS
	multi_critical::compute_h1_timer.start();
#endif

	if(use_logpath) {
	    
	    if(multi_critical::verbose) std::cout << "Now compute h1" << std::endl;
	    
	    for(int ell=L;ell>=3;ell--) {
		if(multi_critical::very_verbose) std::cout << "Dim=" << ell << std::endl;
		
		relation_sorter.set_dim(ell-2);
		
		long no_elements_with_non_trivial_preimage=0;
		std::vector<index> preimage;
		
		for(Input_simplex& input_simplex : input_complex[ell-1]) {
		    for(int i=0; i<input_simplex.copies.size(); i++) {
			for(auto& p : input_simplex.relations[i]) {
			    int j = p.first;
			    assert(i<j);
			    index relation_idx = p.second;
			    //std::cout << "Handling relation_idx " << relation_idx << std::endl;
			    std::vector<index> image_of_map;
			    test_timer_1.resume();
			    sum_of_composed_maps(f1[ell-1],f1[ell],h0[ell-1],p1[ell],relation_idx,image_of_map);
			    test_timer_1.stop();
			    //compose_maps(f1[ell-1],f1[ell],relation_idx,image_of_map);
			    //compose_maps(h0[ell-1],p1[ell],relation_idx,image_of_map);
			    // Now image_of_map is the sum of the two composed maps.
			    //std::cout << "Size of image is " << image_of_map.size() << std::endl;
			    preimage.clear();
			    test_timer_2.resume();
			    relation_sorter.find_preimage_of_syzygies(image_of_map,preimage);
			    test_timer_2.stop();
			    if(!preimage.empty()) {
				h1[ell][relation_idx]=preimage;
				no_elements_with_non_trivial_preimage++;
			    }
			}
		    }
	
		}
		if(multi_critical::very_verbose) std::cout << no_elements_with_non_trivial_preimage << " elements have non-trivial function value" << std::endl;
	    }
	}

	
#if MULTI_CRITICAL_TIMERS
	multi_critical::compute_h1_timer.stop();
#endif


#if MULTI_CRITICAL_TIMERS
	multi_critical::compute_hh_timer.start();
#endif

	if(use_logpath) {
	    
	    if(multi_critical::verbose) std::cout << "Now compute hh" << std::endl;
	    
	    for(int ell=L-1;ell>=3;ell--) {
		if(multi_critical::very_verbose) std::cout << "Dim=" << ell << std::endl;
		
		relation_sorter.set_dim(ell-2);
		
		long no_elements_with_non_trivial_preimage=0;
		
		for(Input_simplex& input_simplex : input_complex[ell]) {
		    /*
		      {
		      std::cout << "AT: ";
		      for(index idx : input_simplex.boundary) {
		      std::cout << idx << " ";
		      }
		      std::cout << std::endl;
		      }
		    */
		    for(int i=0; i<input_simplex.copies.size(); i++) {
			index gen_idx = input_simplex.copies[i];
			
			std::vector<index> image_of_map;
			sum_of_composed_maps(h0[ell-1],f0[ell],f1[ell-1],h0[ell],gen_idx,image_of_map);
			
			std::vector<index> preimage;
			relation_sorter.find_preimage_of_syzygies(image_of_map,preimage);
			
			if(!preimage.empty()) {
			    hh[ell][gen_idx]=preimage;
			    no_elements_with_non_trivial_preimage++;
			}
		    }
		}
		
		if(multi_critical::very_verbose) std::cout << no_elements_with_non_trivial_preimage << " elements have non-trivial function value" << std::endl;
	    }
	}
#if MULTI_CRITICAL_TIMERS
	multi_critical::compute_hh_timer.stop();
#endif

#if MULTI_CRITICAL_TIMERS
	multi_critical::construct_boundaries_timer.start();
#endif

	if(multi_critical::verbose) std::cout << "Construct the boundaries" << std::endl;
	
	// Now construct the boundaries of the chain complex

	for(int ell=L+1;ell>=0;ell--) {
	    for(index idx=0;idx<output_complex[ell].size();idx++) {
		Output_simplex& output_simplex = output_complex[ell][idx];
		assert(output_simplex.boundary.empty());
		//std::cout << "ell=" << ell << ", idx=" << idx << ", type:  " << output_simplex.type << " ";
		if(output_simplex.type==GENERATOR) {
		    std::copy(f0[ell][idx].begin(),f0[ell][idx].end(),std::back_inserter(output_simplex.boundary));
		    std::copy(h0[ell][idx].begin(),h0[ell][idx].end(),std::back_inserter(output_simplex.boundary));
		    std::copy(hh[ell][idx].begin(),hh[ell][idx].end(),std::back_inserter(output_simplex.boundary));
		}
		if(output_simplex.type==RELATION) {
		    std::copy(p1[ell][idx].begin(),p1[ell][idx].end(),std::back_inserter(output_simplex.boundary));
		    std::copy(f1[ell][idx].begin(),f1[ell][idx].end(),std::back_inserter(output_simplex.boundary));
		    std::copy(h1[ell][idx].begin(),h1[ell][idx].end(),std::back_inserter(output_simplex.boundary));
		}
		if(output_simplex.type==SYZYGY) {
		    std::copy(p2[ell][idx].begin(),p2[ell][idx].end(),std::back_inserter(output_simplex.boundary));
		    std::copy(f2[ell][idx].begin(),f2[ell][idx].end(),std::back_inserter(output_simplex.boundary));
		    // Map to gen is 0
		}
		/*
		std::cout << "Bd: ";
		for(index p : output_simplex.boundary) {
		    std::cout << p << " ";
		}
		std::cout << std::endl;
		*/
	    }
	}

#if MULTI_CRITICAL_TIMERS
	multi_critical::construct_boundaries_timer.stop();
#endif

	
	// Output complex is done - check whether it is a graded cell complex

#if !NDEBUG
	if(multi_critical::verbose) std::cout << "Checking.." << std::endl;
	check_output_complex(output_complex);

	if(multi_critical::verbose) std::cout << "done checking" << std::endl;
#endif

	// ...and convert to graded matrices

	/* Old code with explizit conversion to pre columns
#if MULTI_CRITICAL_TIMERS
	multi_critical::convert_to_precolumns_timer.start();
#endif
	std::cout << "Convert to pre_columns.." << std::endl;

	typedef typename GradedMatrix::Grade Matrix_Grade;

	
	typedef mpp_utils::Pre_column_struct<Matrix_Grade> Pre_column;
	std::vector<std::vector<Pre_column>> pre_columns;
	pre_columns.resize(L+1);

	for(int ell=L;ell>=0;ell--) {
	    for(int idx=0;idx<output_complex[ell].size();idx++) {
		Output_simplex& output_simplex=output_complex[ell][idx];
		Matrix_Grade gr(output_simplex.grade.x,output_simplex.grade.y);
		pre_columns[L-ell].push_back(Pre_column(idx,gr,output_simplex.boundary));
	    }
	}
	mpp_utils::create_graded_matrices_from_pre_column_struct(pre_columns,result,0,false,false);
	
#endif
#if MULTI_CRITICAL_TIMERS
	multi_critical::convert_to_precolumns_timer.stop();
#endif
	*/

#if MULTI_CRITICAL_TIMERS
	multi_critical::sort_by_grades_timer.start();
#endif

	std::vector<std::vector<index>> permutations, inv_permutations;
	permutations.resize(output_complex.size());
	inv_permutations.resize(output_complex.size());
	for(int i=0;i<output_complex.size();i++) {
	    long m = output_complex[i].size();
	    permutations[i].resize(m);
	    inv_permutations[i].resize(m);
	    for(int j=0;j<m;j++) {
		permutations[i][j]=j;
	    }
#if 1
	    Grade_sorter sorter(output_complex[i]);
	    std::sort(permutations[i].begin(),permutations[i].end(),sorter);
#endif
	    for(int j=0;j<m;j++) {
		inv_permutations[i][permutations[i][j]]=j;
	    }
	}

#if MULTI_CRITICAL_TIMERS
	multi_critical::sort_by_grades_timer.stop();
#endif
	
	
#if MULTI_CRITICAL_TIMERS
	multi_critical::convert_to_graded_matrices_timer.start();
#endif
	if(multi_critical::verbose) std::cout << "Convert to graded matrices" << std::endl;
	typedef typename GradedMatrix::Grade Matrix_Grade;
	Output_complex_accessor<Matrix_Grade> accessor(output_complex,permutations,inv_permutations);
	mpp_utils::create_graded_matrices_from_pre_column_struct(accessor,result,0,false);

#if !NDEBUG
	for(int i=0;i<result.size();i++) {
	    assert(mpp_utils::is_lex_sorted(result[i]));
	}
#endif
	
#if MULTI_CRITICAL_TIMERS
	multi_critical::convert_to_graded_matrices_timer.stop();
#endif


	
	return;
    }
    
    
}
