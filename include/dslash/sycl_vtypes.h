#pragma once

#include <cassert>
#include "lattice/constants.h"
#include "dslash/dslash_defaults.h"

#include "lattice/lattice_info.h"
#include "dslash/dslash_complex.h"
#include "dslash/dslash_scalar_complex_ops.h"
#include "dslash/dslash_vectype_sycl.h"
#include "dslash/sycl_view.h"
#include "dslash/dslash_vnode.h"
#include "dslash/sycl_vneighbor_table.h"


namespace MG {
 IndexArray block(IndexArray input, IndexArray block_factors){
	 IndexArray ret_val = input;
	 for(int mu=0; mu < 4; ++mu ) {
		 assert( ret_val[mu] % block_factors[mu] == 0);
		 ret_val[mu]/= block_factors[mu];
	 }
     return ret_val;
 }


 template<typename T, typename VN, int _num_spins>
 class SyCLCBFineVSpinor {
 public:

	   using VecType = SIMDComplexSyCL<typename BaseType<T>::Type, VN::VecLen>;
	   using DataType = View<VecType,3,DefaultSpinorLayout>;

	   template<cl::sycl::access::mode accessMode, cl::sycl::access::target accessTarget = cl::sycl::access::target::global_buffer>
	   using DataAccessor = ViewAccessor<VecType,3,DefaultSpinorLayout,accessMode,accessTarget>;

	   SyCLCBFineVSpinor(const LatticeInfo& info, IndexType cb)
	   	   : _g_info(info), _cb(cb), _info(block(info.GetLatticeOrigin(),{ VN::Dim0, VN::Dim1, VN::Dim2, VN::Dim3 }),
		   	   	   	   	   	       block(info.GetLatticeDimensions(),{ VN::Dim0, VN::Dim1, VN::Dim2, VN::Dim3 }),
								   info.GetNumSpins(),
								   info.GetNumColors(),
								   info.GetNodeInfo()),
								   _cb_data("cb_data", {_info.GetNumCBSites(),_num_spins,3}) {


     if( _g_info.GetNumColors() != 3 ) {
       MasterLog(ERROR, "CBFineSpinor has to have 3 colors in info. Info has %d", _g_info.GetNumColors());
     }
     if( _g_info.GetNumSpins() != _num_spins ) {
       MasterLog(ERROR, "CBFineSpinor has to have %d spins in info. Info has %d",
		 _num_spins,_g_info.GetNumSpins());
     }
  }


   inline
     const LatticeInfo& GetInfo() const {
     return (_info);
   }

   inline 
     const LatticeInfo& GetGlobalInfo() const {
     return _g_info;
   }


   inline
     IndexType GetCB() const {
     return _cb;
   }




   DataType GetData() const {
     return _cb_data;
   }
  
   DataType& GetData() {
     return _cb_data;
   }

   
 private:
   const LatticeInfo& _g_info;
   const IndexType _cb;
   LatticeInfo _info;
   DataType _cb_data;

 };

 
 template<typename T, typename VN>
   using SyCLVSpinorView =  typename SyCLCBFineVSpinor<T,VN,4>::DataType;

 template<typename T, typename VN, cl::sycl::access::mode accessMode, cl::sycl::access::target accessTarget = cl::sycl::access::target::global_buffer>
   using SyCLVSpinorViewAccessor = typename SyCLCBFineVSpinor<T,VN,4>::template DataAccessor<accessMode,accessTarget>;

 template<typename T, typename VN>
   using SyCLVHalfSpinorView =  typename SyCLCBFineVSpinor<T,VN,2>::DataType;

 template<typename T, typename VN, cl::sycl::access::mode accessMode, cl::sycl::access::target accessTarget = cl::sycl::access::target::global_buffer>
   using SyCLVHalfSpinorViewAccessor = typename SyCLCBFineVSpinor<T,VN,2>::template DataAccessor<accessMode,accessTarget>;

 /// SG Spinor

 template<typename T, typename VN, int _num_spins>
  class SyCLCBFineSGVSpinor {
  public:

	   using ScalarType = typename BaseType<T>::Type;
	   using DataType = View<ScalarType,5,LayoutRight>;   // Always layout right

 	   template<cl::sycl::access::mode accessMode,
	   	       cl::sycl::access::target accessTarget = cl::sycl::access::target::global_buffer>
 	   using DataAccessor = ViewAccessor<ScalarType,5,LayoutRight,accessMode,accessTarget>;

 	   SyCLCBFineSGVSpinor(const LatticeInfo& info, IndexType cb)
 	   	   : _g_info(info), _cb(cb), _info(block(info.GetLatticeOrigin(),{ VN::Dim0, VN::Dim1, VN::Dim2, VN::Dim3 }),
 		   	   	   	   	   	       block(info.GetLatticeDimensions(),{ VN::Dim0, VN::Dim1, VN::Dim2, VN::Dim3 }),
 								   info.GetNumSpins(),
 								   info.GetNumColors(),
 								   info.GetNodeInfo()),
 								   _cb_data("cb_data", {_info.GetNumCBSites(),_num_spins,3,2,VN::VecLen}) {


      if( _g_info.GetNumColors() != 3 ) {
        MasterLog(ERROR, "CBFineSpinor has to have 3 colors in info. Info has %d", _g_info.GetNumColors());
      }
      if( _g_info.GetNumSpins() != _num_spins ) {
        MasterLog(ERROR, "CBFineSpinor has to have %d spins in info. Info has %d",
 		 _num_spins,_g_info.GetNumSpins());
      }
   }


    inline
      const LatticeInfo& GetInfo() const {
      return (_info);
    }

    inline
      const LatticeInfo& GetGlobalInfo() const {
      return _g_info;
    }


    inline
      IndexType GetCB() const {
      return _cb;
    }




    DataType GetData() const {
      return _cb_data;
    }

    DataType& GetData() {
      return _cb_data;
    }


  private:
    const LatticeInfo& _g_info;
    const IndexType _cb;
    LatticeInfo _info;
    DataType _cb_data;

  };


  template<typename T, typename VN>
    using SyCLSGVSpinorView =  typename SyCLCBFineSGVSpinor<T,VN,4>::DataType;

  template<typename T, typename VN, cl::sycl::access::mode accessMode, cl::sycl::access::target accessTarget = cl::sycl::access::target::global_buffer>
    using SyCLSGVSpinorViewAccessor = typename SyCLCBFineSGVSpinor<T,VN,4>::template DataAccessor<accessMode,accessTarget>;

  template<typename T, typename VN>
    using SyCLSGVHalfSpinorView =  typename SyCLCBFineSGVSpinor<T,VN,2>::DataType;

  template<typename T, typename VN, cl::sycl::access::mode accessMode, cl::sycl::access::target accessTarget = cl::sycl::access::target::global_buffer>
    using SyCLSGVHalfSpinorViewAccessor = typename SyCLCBFineSGVSpinor<T,VN,2>::template DataAccessor<accessMode,accessTarget>;


 ///--------- Gauge ------------
 template<typename T, typename VN>
 class SyCLCBFineVGaugeField {
 public:

	   using VecType = SIMDComplexSyCL<typename BaseType<T>::Type, VN::VecLen>;
	   using DataType = View<VecType,4,DefaultGaugeLayout>;

	   template<cl::sycl::access::mode accessMode, cl::sycl::access::target accessTarget = cl::sycl::access::target::global_buffer>
	   using DataAccessor = ViewAccessor<VecType,4,DefaultGaugeLayout,accessMode,accessTarget>;


 SyCLCBFineVGaugeField(const LatticeInfo& info, IndexType cb)
   : _g_info(info), _cb(cb), _info(block(info.GetLatticeOrigin(),{ VN::Dim0, VN::Dim1, VN::Dim2, VN::Dim3 }),
 	   	   	       block(info.GetLatticeDimensions(),{ VN::Dim0, VN::Dim1, VN::Dim2, VN::Dim3 }),
		   info.GetNumSpins(),
		   info.GetNumColors(),
		   info.GetNodeInfo()),
		   _cb_data("cb_data", {_info.GetNumCBSites(), 4,3,3}) {

     if( _g_info.GetNumColors() != 3 ) {
       MasterLog(ERROR, "KokkosCBFineSpinor has to have 3 colors in info. Info has %d", _g_info.GetNumColors());
     }
    
		   }

   inline
     const LatticeInfo& GetInfo() const {
     return (_info);
   }

   inline 
     const LatticeInfo& GetGlobalInfo() const {
     return _g_info;
   }


   inline
     IndexType GetCB() const {
     return _cb;
   }


   const DataType& GetData() const {
     return (*this)._cb_data;
   }
  

   DataType& GetData() {
     return (*this)._cb_data;
   }

   
 private:

   const LatticeInfo& _g_info;
   LatticeInfo _info;
   DataType _cb_data;
   const IndexType _cb;
 };

 template<typename T, typename VN>
   class SyCLFineVGaugeField {
 private:
   const LatticeInfo& _info;
   SyCLCBFineVGaugeField<T,VN>  _gauge_data_even;
   SyCLCBFineVGaugeField<T,VN>  _gauge_data_odd;
 public:
   SyCLFineVGaugeField(const LatticeInfo& info) :  _info(info), _gauge_data_even(info,EVEN), _gauge_data_odd(info,ODD) {
		}

   const SyCLCBFineVGaugeField<T,VN>& operator()(IndexType cb) const
     {
       return  (cb == EVEN) ? _gauge_data_even : _gauge_data_odd;
       //return *(_gauge_data[cb]);
     }
   
   SyCLCBFineVGaugeField<T,VN>& operator()(IndexType cb) {
     return (cb == EVEN) ? _gauge_data_even : _gauge_data_odd;
     //return *(_gauge_data[cb]);
   }
 };

 // Double copied gauge field.






 template<typename T, typename VN>
 class SyCLCBFineVGaugeFieldDoubleCopy {
 public:
 SyCLCBFineVGaugeFieldDoubleCopy(const LatticeInfo& info, IndexType cb)
   : _g_info(info), _cb(cb), _info(block(info.GetLatticeOrigin(),{ VN::Dim0, VN::Dim1, VN::Dim2, VN::Dim3 }),
 	   	   	       block(info.GetLatticeDimensions(),{ VN::Dim0, VN::Dim1, VN::Dim2, VN::Dim3 }),
				   info.GetNumSpins(),
				   info.GetNumColors(),
				   info.GetNodeInfo()),
				   _cb_data("cb_data", {_info.GetNumCBSites(),8,3,3}) {

     if( _g_info.GetNumColors() != 3 ) {
       MasterLog(ERROR, "KokkosCBFineSpinor has to have 3 colors in info. Info has %d", _g_info.GetNumColors());
     }
    
     
     MasterLog(INFO, "Exiting Constructor");
   }


   inline
     const LatticeInfo& GetInfo() const {
     return (_info);
   }

   inline 
     const LatticeInfo& GetGlobalInfo() const {
     return _g_info;
   }


   inline
     IndexType GetCB() const {
     return _cb;
   }

   // Double Copied Gauge Field.
   using VecType = SIMDComplexSyCL<typename BaseType<T>::Type, VN::VecLen>;
   using DataType = View<VecType,4,DefaultGaugeLayout>;

   template<cl::sycl::access::mode accessMode, cl::sycl::access::target accessTarget = cl::sycl::access::target::global_buffer>
   using DataAccessor = ViewAccessor<VecType,4,DefaultGaugeLayout,accessMode,accessTarget>;

   const DataType& GetData() const {
     return _cb_data;
   }
  
   DataType& GetData() {
     return _cb_data;
   }

   void import(const SyCLCBFineVGaugeField<T,VN>& src_cb,
	       const SyCLCBFineVGaugeField<T,VN>& src_othercb)
   {
     using InputType = typename SyCLCBFineVGaugeField<T,VN>::DataType;

     using FType = typename BaseType<T>::Type;

     // Sanity: src_cb has to match my CB
     if(  GetCB() != src_cb.GetCB() ) {
       MasterLog(ERROR, "cb of src_cb does not match my cb in import()");
     }

     // Sanity 2: othercb has to be the opposite CB from me.
     int expected_othercb = (src_cb.GetCB() == EVEN) ? ODD :EVEN;
     if( expected_othercb != src_othercb.GetCB() ) { 
       MasterLog(ERROR, "cb of src_othercb is not opposite of mine in import()");
     }

     // Grab a site table

     IndexArray cb_latdims = _info.GetCBLatticeDimensions();
     SiteTable neigh_table_tab(cb_latdims[0],cb_latdims[1],cb_latdims[2], cb_latdims[3]);
     SiteTableAccess neigh_table=neigh_table_tab.template get_access<cl::sycl::access::mode::read>();
     size_t num_cbsites = _info.GetNumCBSites();
     
     InputType cb_data_in = src_cb.GetData().template get_access<cl::sycl::access::mode::read>();
     InputType othercb_data_in = src_othercb.GetData().template get_access<cl::sycl::access::mode::read>();
     int target_cb = _cb;

#pragma omp parallel for
     for(size_t site = 0; site < num_cbsites; ++site) {
    	 IndexArray site_coords = LayoutLeft::coords(site,cb_latdims);
    	 std::size_t xcb = site_coords[0];
    	 std::size_t y = site_coords[1];
    	 std::size_t z = site_coords[2];
    	 std::size_t t = site_coords[3];

    	 size_t n_idx;
    	 bool do_permute;

    	 // T_minus
    	 neigh_table.NeighborTMinus(xcb,y,z,t,n_idx,do_permute);

    	 if( do_permute ) {
    		 for(int col=0; col < 3; ++col) {
    			 for(int col2=0; col2 < 3; ++col2) {
    				 ComplexCopy<FType,VN::VecLen>(_cb_data(site,0,col,col2),VN::permuteT(othercb_data_in(n_idx,3,col,col2)));
    			 }
    		 }
    	 }
    	 else {
    		 for(int col=0; col < 3; ++col) {
    			 for(int col2=0; col2 < 3; ++col2) {
    				 ComplexCopy<FType,VN::VecLen>(_cb_data(site,0,col,col2), othercb_data_in(n_idx,3,col,col2));
    			 }
    		 }


    	 }

    	 // Z_minus
    	 neigh_table.NeighborZMinus(xcb,y,z,t,n_idx,do_permute);
    	 if( do_permute ) {
    		 for(int col=0; col < 3; ++col) {
    			 for(int col2=0; col2 < 3; ++col2) {
    				 ComplexCopy<FType,VN::VecLen>(_cb_data(site,1,col,col2),VN::permuteZ(othercb_data_in(n_idx,2,col,col2)));
    			 }
    		 }
    	 }
    	 else {
    		 for(int col=0; col < 3; ++col) {
    			 for(int col2=0; col2 < 3; ++col2) {
    				 ComplexCopy<FType,VN::VecLen>(_cb_data(site,1,col,col2), othercb_data_in(n_idx,2,col,col2));
    			 }
    		 }
    	 }

    	 // Y_minus
    	 neigh_table.NeighborYMinus(xcb,y,z,t,n_idx,do_permute);
    	 if( do_permute ) {
    		 for(int col=0; col < 3; ++col) {
    			 for(int col2=0; col2 < 3; ++col2) {
    				 ComplexCopy<FType,VN::VecLen>(_cb_data(site,2,col,col2),VN::permuteY(othercb_data_in(n_idx,1,col,col2)));
    			 }
    		 }
    	 }
    	 else {
    		 for(int col=0; col < 3; ++col) {
    			 for(int col2=0; col2 < 3; ++col2) {
    				 ComplexCopy<FType,VN::VecLen>(_cb_data(site,2,col,col2),othercb_data_in(n_idx,1,col,col2));
    			 }
    		 }
    	 }


    	 // X_minus
    	 neigh_table.NeighborXMinus(xcb,y,z,t,target_cb,n_idx,do_permute);
    	 if( do_permute ) {
    		 for(int col=0; col < 3; ++col) {
    			 for(int col2=0; col2 < 3; ++col2) {
    				 ComplexCopy<FType,VN::VecLen>(_cb_data(site,3,col,col2),VN::permuteX(othercb_data_in(n_idx,0,col,col2)));
    			 }
    		 }
    	 }
    	 else {
    		 for(int col=0; col < 3; ++col) {
    			 for(int col2=0; col2 < 3; ++col2) {
    				 ComplexCopy<FType,VN::VecLen>(_cb_data(site,3,col,col2),othercb_data_in(n_idx,0,col,col2));
    			 }
    		 }
    	 }

    	 // X-plus
		 for(int col=0; col < 3; ++col) {
			 for(int col2=0; col2 < 3; ++col2) {
				 ComplexCopy<FType,VN::VecLen>(_cb_data(site,4,col,col2),cb_data_in(site,0,col,col2));
			 }
		 }

		 // Y_Plus
		 for(int col=0; col < 3; ++col) {
			 for(int col2=0; col2 < 3; ++col2) {
				 ComplexCopy<FType,VN::VecLen>(_cb_data(site,5,col,col2),cb_data_in(site,1,col,col2));
			 }
		 }

		 // Z_Plus
		 for(int col=0; col < 3; ++col) {
			 for(int col2=0; col2 < 3; ++col2) {
				 ComplexCopy<FType,VN::VecLen>(_cb_data(site,6,col,col2),cb_data_in(site,2,col,col2));
			 }
		 }

		 // T_Plus
		 for(int col=0; col < 3; ++col) {
			 for(int col2=0; col2 < 3; ++col2) {
				 ComplexCopy<FType,VN::VecLen>(_cb_data(site,7,col,col2),cb_data_in(site,3,col,col2));
			 }
		 }

     }
   }


   

 private:

   const LatticeInfo& _g_info;
   LatticeInfo _info;
   DataType _cb_data;
   const IndexType _cb;
 };

 template<typename T, typename VN>
 void import(SyCLCBFineVGaugeFieldDoubleCopy<T,VN>& target,
		 SyCLCBFineVGaugeField<T,VN> src_cb,
		 SyCLCBFineVGaugeField<T,VN> src_othercb)
 {
	 using InputType = typename SyCLCBFineVGaugeField<T,VN>::DataType;
	 using FType=typename BaseType<T>::Type;

	 // Sanity: src_cb has to match my CB
	 if(  target.GetCB() != src_cb.GetCB() ) {
		 MasterLog(ERROR, "cb of src_cb does not match my cb in import()");
	 }

	 // Sanity 2: othercb has to be the opposite CB from me.
	 int expected_othercb = (src_cb.GetCB() == EVEN) ? ODD :EVEN;
	 if( expected_othercb != src_othercb.GetCB() ) {
		 MasterLog(ERROR, "cb of src_othercb is not opposite of mine in import()");
	 }

	 // Grab a site table
	 const LatticeInfo&  info = target.GetInfo();
	 IndexArray cb_latdims = info.GetCBLatticeDimensions();
	 MasterLog(INFO, "Double Storing Gauge: Info has size=(%d,%d,%d,%d)",
			 cb_latdims[0],cb_latdims[1],cb_latdims[2],cb_latdims[3]);

	 SiteTable neigh_table_tab(cb_latdims[0],cb_latdims[1],cb_latdims[2],
			 cb_latdims[3]);

	 auto neigh_table = neigh_table_tab.template get_access<cl::sycl::access::mode::read>();
	 auto cb_data_out = target.GetData().template get_access<cl::sycl::access::mode::write>();
	 auto cb_data_in = src_cb.GetData().template get_access<cl::sycl::access::mode::read>();
	 auto othercb_data_in= src_othercb.GetData().template get_access<cl::sycl::access::mode::read>();

	 int target_cb = target.GetCB(); // Lambd cannot access member _cb on host
	 int num_cbsites = info.GetNumCBSites();


#pragma omp parallel for
	 for(size_t site=0; site < num_cbsites; ++site) {
		 IndexArray coord_array = LayoutLeft::coords(site,cb_latdims);
		 const size_t xcb = coord_array[0];
		 const size_t y=coord_array[1];
		 const size_t z=coord_array[2];
		 const size_t t=coord_array[3];

		 size_t n_idx;
		 bool do_permute;

		 // T_minus
		 neigh_table.NeighborTMinus(xcb,y,z,t,n_idx,do_permute);

		 if( do_permute ) {
			 for(int col=0; col < 3; ++col) {
				 for(int col2=0; col2 < 3; ++col2) {
					 ComplexCopy<FType,VN::VecLen>(cb_data_out(site,0,col,col2),VN::permuteT(othercb_data_in(n_idx,3,col,col2)));
				 }
			 }
		 }
		 else {
			 for(int col=0; col < 3; ++col) {
				 for(int col2=0; col2 < 3; ++col2) {
					 ComplexCopy<FType,VN::VecLen>(cb_data_out(site,0,col,col2), othercb_data_in(n_idx,3,col,col2));
					 //cb_data_out(site,0,col,col2) = VN::permute(mask, othercb_data_in(n_idx,3,col,col2));
				 }
			 }
		 }
		 // Z_minus

		 neigh_table.NeighborZMinus(xcb,y,z,t,n_idx,do_permute);
		 if( do_permute ) {

			 for(int col=0; col < 3; ++col) {
				 for(int col2=0; col2 < 3; ++col2) {
					 ComplexCopy<FType,VN::VecLen>(cb_data_out(site,1,col,col2),VN::permuteZ(othercb_data_in(n_idx,2,col,col2)));
					 //cb_data_out(site,1,col,col2) = VN::permute(mask, othercb_data_in(n_idx,2,col,col2));
				 }
			 }
		 }
		 else {
			 for(int col=0; col < 3; ++col) {
				 for(int col2=0; col2 < 3; ++col2) {
					 ComplexCopy<FType,VN::VecLen>(cb_data_out(site,1,col,col2),othercb_data_in(n_idx,2,col,col2));
					 //cb_data_out(site,1,col,col2) = VN::permute(mask, othercb_data_in(n_idx,2,col,col2));
				 }
			 }

		 }

		 // Y_minus
		 neigh_table.NeighborYMinus(xcb,y,z,t,n_idx,do_permute);

		 if( do_permute ) {
			 for(int col=0; col < 3; ++col) {
				 for(int col2=0; col2 < 3; ++col2) {
					 ComplexCopy<FType,VN::VecLen>(cb_data_out(site,2,col,col2),VN::permuteY(othercb_data_in(n_idx,1,col,col2)));
					 //cb_data_out(site,2,col,col2) = VN::permute(mask, othercb_data_in(n_idx,1,col,col2));
				 }
			 }
		 }
		 else {
			 for(int col=0; col < 3; ++col) {
				 for(int col2=0; col2 < 3; ++col2) {
					 ComplexCopy<FType,VN::VecLen>(cb_data_out(site,2,col,col2),othercb_data_in(n_idx,1,col,col2));
					 //cb_data_out(site,2,col,col2) = VN::permute(mask, othercb_data_in(n_idx,1,col,col2));
				 }
			 }
		 }
		 // X_minus

		 neigh_table.NeighborXMinus(xcb,y,z,t,target_cb,n_idx,do_permute);

		 if( do_permute ) {
			 for(int col=0; col < 3; ++col) {
				 for(int col2=0; col2 < 3; ++col2) {
					 ComplexCopy<FType,VN::VecLen>(cb_data_out(site,3,col,col2),VN::permuteX(othercb_data_in(n_idx,0,col,col2)));
					 //cb_data_out(site,3,col,col2) = VN::permute(mask, othercb_data_in(n_idx,0,col,col2));
				 }
			 }
		 }
		 else {
			 for(int col=0; col < 3; ++col) {
				 for(int col2=0; col2 < 3; ++col2) {
					 ComplexCopy<FType,VN::VecLen>(cb_data_out(site,3,col,col2),othercb_data_in(n_idx,0,col,col2));
					 //cb_data_out(site,3,col,col2) = VN::permute(mask, othercb_data_in(n_idx,0,col,col2));
				 }
			 }

		 }

		 // X-Plus
		 for(int col=0; col < 3; ++col) {
			 for(int col2=0; col2 < 3; ++col2) {
				 ComplexCopy<FType,VN::VecLen>(cb_data_out(site,4,col,col2),cb_data_in(site,0,col,col2));
				 //cb_data_out(site,4,col,col2)=cb_data_in(site,0,col,col2);
			 }
		 }

		 // Y_Plus
		 for(int col=0; col < 3; ++col) {
			 for(int col2=0; col2 < 3; ++col2) {
				 ComplexCopy<FType,VN::VecLen>(cb_data_out(site,5,col,col2),cb_data_in(site,1,col,col2));
				 //cb_data_out(site,5,col,col2)=cb_data_in(site,1,col,col2);
			 }
		 }

		 // Z_Plus
		 for(int col=0; col < 3; ++col) {
			 for(int col2=0; col2 < 3; ++col2) {
				 ComplexCopy<FType,VN::VecLen>(cb_data_out(site,6,col,col2),cb_data_in(site,2,col,col2));
				 //cb_data_out(site,6,col,col2)=cb_data_in(site,2,col,col2);
			 }
		 }

		 // T_Plus
		 for(int col=0; col < 3; ++col) {
			 for(int col2=0; col2 < 3; ++col2) {
				 ComplexCopy<FType,VN::VecLen>(cb_data_out(site,7,col,col2),cb_data_in(site,3,col,col2));
				 //cb_data_out(site,7,col,col2)=cb_data_in(site,3,col,col2);
			 }
		 }

	 }// parallel for



 }


 template<typename T,typename VN>
   using SyCLVGaugeView = typename SyCLCBFineVGaugeFieldDoubleCopy<T,VN>::DataType;

 template<typename T, typename VN, cl::sycl::access::mode accessMode, cl::sycl::access::target accessTarget = cl::sycl::access::target::global_buffer>
    using SyCLVGaugeViewAccessor = typename  SyCLCBFineVGaugeFieldDoubleCopy<T,VN>::template DataAccessor<accessMode,accessTarget>;





 ///--------- Subgroup Gauge ------------
 template<typename T, typename VN>
 class SyCLCBFineSGVGaugeField {
 public:

	   using ScalarType = typename BaseType<T>::Type;

	   using DataType = View<ScalarType,6,LayoutRight>;

	   template<cl::sycl::access::mode accessMode, cl::sycl::access::target accessTarget = cl::sycl::access::target::global_buffer>
	   using DataAccessor = ViewAccessor<ScalarType,6,LayoutRight,accessMode,accessTarget>;


 SyCLCBFineSGVGaugeField(const LatticeInfo& info, IndexType cb)
   : _g_info(info), _cb(cb), _info( block(info.GetLatticeOrigin(),std::array<size_t,4>{ VN::Dim0, VN::Dim1, VN::Dim2, VN::Dim3 }),
 	   	   	       block(info.GetLatticeDimensions(),std::array<size_t,4>{ VN::Dim0, VN::Dim1, VN::Dim2, VN::Dim3 }),
		   info.GetNumSpins(),
		   info.GetNumColors(),
		   info.GetNodeInfo()),
		   _cb_data("cb_data", {_info.GetNumCBSites(), 4,3,3,2,VN::VecLen}) {

     if( _g_info.GetNumColors() != 3 ) {
       MasterLog(ERROR, "KokkosCBFineSpinor has to have 3 colors in info. Info has %d", _g_info.GetNumColors());
     }

		   }

   inline
     const LatticeInfo& GetInfo() const {
     return (_info);
   }

   inline
     const LatticeInfo& GetGlobalInfo() const {
     return _g_info;
   }


   inline
     IndexType GetCB() const {
     return _cb;
   }


   const DataType& GetData() const {
     return (*this)._cb_data;
   }


   DataType& GetData() {
     return (*this)._cb_data;
   }


 private:

   const LatticeInfo& _g_info;
   LatticeInfo _info;
   DataType _cb_data;
   const IndexType _cb;
 };

 template<typename T, typename VN>
   class SyCLFineSGVGaugeField {
 private:
   const LatticeInfo& _info;
   SyCLCBFineSGVGaugeField<T,VN>  _gauge_data_even;
   SyCLCBFineSGVGaugeField<T,VN>  _gauge_data_odd;
 public:
   SyCLFineSGVGaugeField(const LatticeInfo& info) :  _info(info), _gauge_data_even(info,EVEN), _gauge_data_odd(info,ODD) {
		}

   const SyCLCBFineSGVGaugeField<T,VN>& operator()(IndexType cb) const
     {
       return  (cb == EVEN) ? _gauge_data_even : _gauge_data_odd;
       //return *(_gauge_data[cb]);
     }

   SyCLCBFineSGVGaugeField<T,VN>& operator()(IndexType cb) {
     return (cb == EVEN) ? _gauge_data_even : _gauge_data_odd;
     //return *(_gauge_data[cb]);
   }
 };

 // Double copied gauge field.






 template<typename T, typename VN>
 class SyCLCBFineSGVGaugeFieldDoubleCopy {
 public:
 SyCLCBFineSGVGaugeFieldDoubleCopy(const LatticeInfo& info, IndexType cb)
   : _g_info(info), _cb(cb), _info(block(info.GetLatticeOrigin(),{ VN::Dim0, VN::Dim1, VN::Dim2, VN::Dim3 }),
 	   	   	       block(info.GetLatticeDimensions(),{ VN::Dim0, VN::Dim1, VN::Dim2, VN::Dim3 }),
				   info.GetNumSpins(),
				   info.GetNumColors(),
				   info.GetNodeInfo()),
				   _cb_data("cb_data", {_info.GetNumCBSites(),8,3,3,2,VN::VecLen}) {

     if( _g_info.GetNumColors() != 3 ) {
       MasterLog(ERROR, "KokkosCBFineSpinor has to have 3 colors in info. Info has %d", _g_info.GetNumColors());
     }


     MasterLog(INFO, "Exiting Constructor");
   }


   inline
     const LatticeInfo& GetInfo() const {
     return (_info);
   }

   inline
     const LatticeInfo& GetGlobalInfo() const {
     return _g_info;
   }


   inline
     IndexType GetCB() const {
     return _cb;
   }

   // Double Copied Gauge Field.
   using ScalarType = typename BaseType<T>::Type;
   using DataType = View<ScalarType,6,LayoutRight>;

   template<cl::sycl::access::mode accessMode, cl::sycl::access::target accessTarget = cl::sycl::access::target::global_buffer>
   using DataAccessor = ViewAccessor<ScalarType,6,LayoutRight,accessMode,accessTarget>;

   const DataType& GetData() const {
     return _cb_data;
   }

   DataType& GetData() {
     return _cb_data;
   }

   void import(const SyCLCBFineSGVGaugeField<T,VN>& src_cb,
	       const SyCLCBFineSGVGaugeField<T,VN>& src_othercb)
   {
     using InputType = typename SyCLCBFineSGVGaugeField<T,VN>::DataType;

     using FType = typename BaseType<T>::Type;

     // Sanity: src_cb has to match my CB
     if(  GetCB() != src_cb.GetCB() ) {
       MasterLog(ERROR, "cb of src_cb does not match my cb in import()");
     }

     // Sanity 2: othercb has to be the opposite CB from me.
     int expected_othercb = (src_cb.GetCB() == EVEN) ? ODD :EVEN;
     if( expected_othercb != src_othercb.GetCB() ) {
       MasterLog(ERROR, "cb of src_othercb is not opposite of mine in import()");
     }

     // Grab a site table

     IndexArray cb_latdims = _info.GetCBLatticeDimensions();
     SiteTable neigh_table_tab(cb_latdims[0],cb_latdims[1],cb_latdims[2], cb_latdims[3]);
     SiteTableAccess neigh_table=neigh_table_tab.template get_access<cl::sycl::access::mode::read>();
     size_t num_cbsites = _info.GetNumCBSites();

     InputType cb_data_in = src_cb.GetData().template get_access<cl::sycl::access::mode::read>();
     InputType othercb_data_in = src_othercb.GetData().template get_access<cl::sycl::access::mode::read>();
     int target_cb = _cb;

#pragma omp parallel for
     for(size_t site = 0; site < num_cbsites; ++site) {
    	 IndexArray site_coords = LayoutLeft::coords(site,cb_latdims);
    	 std::size_t xcb = site_coords[0];
    	 std::size_t y = site_coords[1];
    	 std::size_t z = site_coords[2];
    	 std::size_t t = site_coords[3];

    	 size_t n_idx;
    	 bool do_permute;

    	 // T_minus
    	 neigh_table.NeighborTMinus(xcb,y,z,t,n_idx,do_permute);
    	 for(int col=0; col < 3; ++col) {
    		 for(int col2=0; col2 < 3; ++col2) {
    			 for(int cmpx=0; cmpx < 2; ++cmpx) {

    				 for(int lane=0; lane < VN::VecLen; ++lane) {
    					 size_t from_lane = do_permute ? VN::t_mask[lane] : lane;

    					 ComplexCopy(_cb_data(site,0,col,col2,cmpx,lane),
    							 othercb_data_in(n_idx,3,col,col2,cmpx,from_lane));
    				 }
    			 }
    		 }
    	 }

    	 // Z_minus
    	 neigh_table.NeighborZMinus(xcb,y,z,t,n_idx,do_permute);
    	 for(int col=0; col < 3; ++col) {
    		 for(int col2=0; col2 < 3; ++col2) {
    			 for(int cmpx=0; cmpx < 2; ++cmpx) {

    				 for(int lane=0; lane < VN::VecLen; ++lane) {
    					 size_t from_lane = do_permute ? VN::z_mask[lane] : lane;

    					 ComplexCopy(_cb_data(site,1,col,col2,cmpx,lane),
    							 othercb_data_in(n_idx,2,col,col2,cmpx,from_lane));
    				 }
    			 }
    		 }
    	 }

    	 // Y minus
    	 neigh_table.NeighborYMinus(xcb,y,z,t,n_idx,do_permute);
    	 for(int col=0; col < 3; ++col) {
    		 for(int col2=0; col2 < 3; ++col2) {
    			 for(int cmpx=0; cmpx < 2; ++cmpx) {

    				 for(int lane=0; lane < VN::VecLen; ++lane) {
    					 size_t from_lane = do_permute ? VN::y_mask[lane] : lane;

    					 ComplexCopy(_cb_data(site,2,col,col2,cmpx,lane),
    							 othercb_data_in(n_idx,1,col,col2,cmpx,from_lane));
    				 }
    			 }
    		 }
    	 }



    	 // X_minus
    	 neigh_table.NeighborXMinus(xcb,y,z,t,target_cb,n_idx,do_permute);
    	 for(int col=0; col < 3; ++col) {
    		 for(int col2=0; col2 < 3; ++col2) {
    			 for(int cmpx=0; cmpx < 2; ++cmpx) {
    				 for(int lane=0; lane < VN::VecLen; ++lane) {
    					 size_t from_lane = do_permute ? VN::x_mask[lane] : lane;

    					 ComplexCopy(_cb_data(site,3,col,col2,cmpx,lane),
    							 othercb_data_in(n_idx,0,col,col2,cmpx,from_lane));
    				 }
    			 }
    		 }
    	 }

    	 // X-plus
    	 for(int col=0; col < 3; ++col) {
    		 for(int col2=0; col2 < 3; ++col2) {
    			 for(int cmpx=0; cmpx < 2; ++cmpx ) {
    				 for(int lane=0; lane < VN::VecLen; ++lane) {
    					 ComplexCopy(_cb_data(site,4,col,col2,cmpx,lane),cb_data_in(site,0,col,col2,cmpx,lane));
    				 }
    			 }
    		 }
    	 }
    	 // Y-plus
    	 for(int col=0; col < 3; ++col) {
    		 for(int col2=0; col2 < 3; ++col2) {
    			 for(int cmpx=0; cmpx < 2; ++cmpx ) {
    				 for(int lane=0; lane < VN::VecLen; ++lane) {
    					 ComplexCopy(_cb_data(site,5,col,col2,cmpx,lane),cb_data_in(site,1,col,col2,cmpx,lane));
    				 }
    			 }
    		 }
    	 }

    	 // Z-plus
    	 for(int col=0; col < 3; ++col) {
    		 for(int col2=0; col2 < 3; ++col2) {
    			 for(int cmpx=0; cmpx < 2; ++cmpx ) {
    				 for(int lane=0; lane < VN::VecLen; ++lane) {
    					 ComplexCopy(_cb_data(site,6,col,col2,cmpx,lane),cb_data_in(site,2,col,col2,cmpx,lane));
    				 }
    			 }
    		 }
    	 }

    	 // T-plus
    	 for(int col=0; col < 3; ++col) {
    		 for(int col2=0; col2 < 3; ++col2) {
    			 for(int cmpx=0; cmpx < 2; ++cmpx ) {
    				 for(int lane=0; lane < VN::VecLen; ++lane) {
    					 ComplexCopy(_cb_data(site,7,col,col2,cmpx,lane),cb_data_in(site,3,col,col2,cmpx,lane));
    				 }
    			 }
    		 }
    	 }




     }
   }




 private:

   const LatticeInfo& _g_info;
   LatticeInfo _info;
   DataType _cb_data;
   const IndexType _cb;
 };

 template<typename T, typename VN>
 void import(SyCLCBFineSGVGaugeFieldDoubleCopy<T,VN>& target,
		 SyCLCBFineSGVGaugeField<T,VN> src_cb,
		 SyCLCBFineSGVGaugeField<T,VN> src_othercb)
 {
	 using InputType = typename SyCLCBFineSGVGaugeField<T,VN>::DataType;
	 using FType=typename BaseType<T>::Type;

	 // Sanity: src_cb has to match my CB
	 if(  target.GetCB() != src_cb.GetCB() ) {
		 MasterLog(ERROR, "cb of src_cb does not match my cb in import()");
	 }

	 // Sanity 2: othercb has to be the opposite CB from me.
	 int expected_othercb = (src_cb.GetCB() == EVEN) ? ODD :EVEN;
	 if( expected_othercb != src_othercb.GetCB() ) {
		 MasterLog(ERROR, "cb of src_othercb is not opposite of mine in import()");
	 }

	 // Grab a site table
	 const LatticeInfo&  info = target.GetInfo();
	 IndexArray cb_latdims = info.GetCBLatticeDimensions();
	 MasterLog(INFO, "Double Storing Gauge: Info has size=(%d,%d,%d,%d)",
			 cb_latdims[0],cb_latdims[1],cb_latdims[2],cb_latdims[3]);

	 SiteTable neigh_table_tab(cb_latdims[0],cb_latdims[1],cb_latdims[2],
			 cb_latdims[3]);

	 auto neigh_table = neigh_table_tab.template get_access<cl::sycl::access::mode::read>();
	 auto cb_data_out = target.GetData().template get_access<cl::sycl::access::mode::write>();
	 auto cb_data_in = src_cb.GetData().template get_access<cl::sycl::access::mode::read>();
	 auto othercb_data_in= src_othercb.GetData().template get_access<cl::sycl::access::mode::read>();

	 int target_cb = target.GetCB(); // Lambd cannot access member _cb on host
	 int num_cbsites = info.GetNumCBSites();


#pragma omp parallel for
	 for(size_t site=0; site < num_cbsites; ++site) {
		 IndexArray coord_array = LayoutLeft::coords(site,cb_latdims);
		 const size_t xcb = coord_array[0];
		 const size_t y=coord_array[1];
		 const size_t z=coord_array[2];
		 const size_t t=coord_array[3];

		 size_t n_idx;
		 bool do_permute;

		 // T_minus
		 neigh_table.NeighborTMinus(xcb,y,z,t,n_idx,do_permute);
		 for(int col=0; col < 3; ++col) {
			 for(int col2=0; col2 < 3; ++col2) {
				 for(int cmpx=0; cmpx < 2; ++cmpx) {
					 for(int lane=0; lane < VN::VecLen; ++lane) {
						 size_t lane_from = do_permute ? VN::t_mask[lane] : lane;

						 cb_data_out(site,0,col,col2,cmpx,lane) = othercb_data_in(n_idx,3,col,col2,cmpx, lane_from);

					 }
				 }
			 }
		 }

		 // Z_minus
		 neigh_table.NeighborZMinus(xcb,y,z,t,n_idx,do_permute);
		 for(int col=0; col < 3; ++col) {
			 for(int col2=0; col2 < 3; ++col2) {
				 for(int cmpx=0; cmpx < 2; ++cmpx) {
					 for(int lane=0; lane < VN::VecLen; ++lane) {
						 size_t lane_from = do_permute ? VN::z_mask[lane] : lane;
						 cb_data_out(site,1,col,col2,cmpx,lane) = othercb_data_in(n_idx,2,col,col2,cmpx, lane_from);



					 }
				 }
			 }
		 }

		 // Y_minus
		 neigh_table.NeighborYMinus(xcb,y,z,t,n_idx,do_permute);
		 for(int col=0; col < 3; ++col) {
			 for(int col2=0; col2 < 3; ++col2) {
				 for(int cmpx=0; cmpx < 2; ++cmpx) {
					 for(int lane=0; lane < VN::VecLen; ++lane) {
						 size_t lane_from = do_permute ? VN::y_mask[lane] : lane;

						 cb_data_out(site,2,col,col2,cmpx,lane) = othercb_data_in(n_idx,1,col,col2,cmpx, lane_from);



					 }
				 }
			 }
		 }

		 // X_minus

		 neigh_table.NeighborXMinus(xcb,y,z,t,target_cb,n_idx,do_permute);
		 for(int col=0; col < 3; ++col) {
			 for(int col2=0; col2 < 3; ++col2) {
				 for(int cmpx=0; cmpx < 2; ++cmpx) {
					 for(int lane=0; lane < VN::VecLen; ++lane) {
						 size_t lane_from = do_permute ? VN::x_mask[lane] : lane;

						 cb_data_out(site,3,col,col2,cmpx,lane) = othercb_data_in(n_idx,0,col,col2,cmpx, lane_from);

					 }
				 }
			 }
		 }

		 // X-Plus
		 for(int col=0; col < 3; ++col) {
			 for(int col2=0; col2 < 3; ++col2) {
				 for(int cmpx=0; cmpx < 2; ++cmpx) {
					 for(int lane=0; lane < VN::VecLen; ++lane) {
						 cb_data_out(site,4,col,col2,cmpx,lane) = cb_data_in(site,0,col,col2,cmpx,lane);
					 }
				 }
			 }
		 }

		 // Y_Plus
		 for(int col=0; col < 3; ++col) {
			 for(int col2=0; col2 < 3; ++col2) {
				 for(int cmpx=0; cmpx < 2; ++cmpx) {
					 for(int lane=0; lane < VN::VecLen; ++lane) {
						 cb_data_out(site,5,col,col2,cmpx,lane) = cb_data_in(site,1,col,col2,cmpx,lane);
					 }
				 }
			 }
		 }

		 // Z_Plus
		 for(int col=0; col < 3; ++col) {
			 for(int col2=0; col2 < 3; ++col2) {
				 for(int cmpx=0; cmpx < 2; ++cmpx) {
					 for(int lane=0; lane < VN::VecLen; ++lane) {
						 cb_data_out(site,6,col,col2,cmpx,lane) = cb_data_in(site,2,col,col2,cmpx,lane);
					 }
				 }
			 }
		 }

		 for(int col=0; col < 3; ++col) {
			 for(int col2=0; col2 < 3; ++col2) {
				 for(int cmpx=0; cmpx < 2; ++cmpx) {
					 for(int lane=0; lane < VN::VecLen; ++lane) {
						 cb_data_out(site,7,col,col2,cmpx,lane) = cb_data_in(site,3,col,col2,cmpx,lane);
					 }
				 }
			 }
		 }

	 }// parallel for



 }


 template<typename T,typename VN>
   using SyCLSGVGaugeView = typename SyCLCBFineSGVGaugeFieldDoubleCopy<T,VN>::DataType;

 template<typename T, typename VN, cl::sycl::access::mode accessMode, cl::sycl::access::target accessTarget = cl::sycl::access::target::global_buffer>
    using SyCLSGVGaugeViewAccessor = typename  SyCLCBFineSGVGaugeFieldDoubleCopy<T,VN>::template DataAccessor<accessMode,accessTarget>;










 // Site views, these are for use inside kernels and should registerize data
	template<typename T,const int S, const int C>
	struct SiteView {
		T _data[S][C];
		T& operator()(int color, int spin) {
			return _data[spin][color];
		}
		const T& operator()(int color, int spin) const {
			return _data[spin][color];
		}
	};

	template<typename T>
	using SpinorSiteView = SiteView<T ,4,3>;

	template<typename T>
	using HalfSpinorSiteView = SiteView< T,2,3>;

	template<typename T>
	  using GaugeSiteView = SiteView< T ,3,3>;
}
