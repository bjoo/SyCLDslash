#ifndef TEST_KOKKOS_VTYPE_H
#define TEST_KOKKOS_VTYPE_H

#include <memory>
#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "dslash_complex.h"
#include "dslash_vectype_sycl.h"
#include "dslash_vnode.h"


#include "kokkos_vneighbor_table.h"

#undef MG_KOKKOS_USE_MDRANGE
namespace MG {

 template<typename T, typename VN, int _num_spins>
 class KokkosCBFineVSpinor {
 public:
 KokkosCBFineVSpinor(const LatticeInfo& info, IndexType cb)
   : _g_info(info), _cb(cb) {

     if( _g_info.GetNumColors() != 3 ) {
       MasterLog(ERROR, "KokkosCBFineSpinor has to have 3 colors in info. Info has %d", _g_info.GetNumColors());
     }
     if( _g_info.GetNumSpins() != _num_spins ) {
       MasterLog(ERROR, "KokkosCBFineSpinor has to have %d spins in info. Info has %d",
		 _num_spins,_g_info.GetNumSpins());
     }

     IndexArray l_orig = _g_info.GetLatticeOrigin();
     IndexArray l_dims = _g_info.GetLatticeDimensions();
     IndexArray VNDims = { VN::Dim0, VN::Dim1, VN::Dim2, VN::Dim3 };
     for(int mu=0; mu < 4; ++mu ) {
       if( l_dims[mu] % VNDims[mu] == 0 ) { 
	 l_orig[mu] /= VNDims[mu];
	 l_dims[mu] /= VNDims[mu];
       }
       else{
	 MasterLog(ERROR, "Local dimension %d (=%d) not divisible by VNode::dims[%d]=%d",
		   mu, l_dims[mu], mu, VNDims[mu]);
       }
     }
     
     _info=std::make_shared<LatticeInfo>(l_orig,l_dims,_g_info.GetNumSpins(), _g_info.GetNumColors(), _g_info.GetNodeInfo());
     
     // Init the data
     _cb_data=DataType("cb_data", _info->GetNumCBSites());
   }


   inline
     const LatticeInfo& GetInfo() const {
     return *(_info);
   }

   inline 
     const LatticeInfo& GetGlobalInfo() const {
     return _g_info;
   }


   inline
     IndexType GetCB() const {
     return _cb;
   }

   using VecType = SIMDComplex<typename BaseType<T>::Type, VN::VecLen>;
   using DataType = Kokkos::View<VecType*[_num_spins][3],Layout,MemorySpace>;


   KOKKOS_INLINE_FUNCTION
   const DataType& GetData() const {
     return _cb_data;
   }
  
   KOKKOS_INLINE_FUNCTION 
   DataType& GetData() {
     return _cb_data;
   }

   
 private:
   DataType _cb_data;
   const LatticeInfo& _g_info;
   std::shared_ptr<LatticeInfo> _info;

   const IndexType _cb;
 };

 
 template<typename T, typename VN>
   using VSpinorView =  typename KokkosCBFineVSpinor<T,VN,4>::DataType;

 template<typename T, typename VN>
   using VHalfSpinorView =  typename KokkosCBFineVSpinor<T,VN,2>::DataType;

 template<typename T, typename VN>
 class KokkosCBFineVGaugeField {
 public:
 KokkosCBFineVGaugeField(const LatticeInfo& info, IndexType cb)
   : _g_info(info), _cb(cb) {

     if( _g_info.GetNumColors() != 3 ) {
       MasterLog(ERROR, "KokkosCBFineSpinor has to have 3 colors in info. Info has %d", _g_info.GetNumColors());
     }
    
     IndexArray l_orig = _g_info.GetLatticeOrigin();
     IndexArray l_dims = _g_info.GetLatticeDimensions();
     IndexArray VNDims = { VN::Dim0, VN::Dim1, VN::Dim2, VN::Dim3};
     for(int mu=0; mu < 4; ++mu ) {
       if( l_dims[mu] % VNDims[mu] == 0 ) { 
	 l_orig[mu] /= VNDims[mu];
	 l_dims[mu] /= VNDims[mu];
       }
       else{
	 MasterLog(ERROR, "Local dimension %d (=%d) not divisible by VNode::dims[%d]=%d",
		   mu, l_dims[mu], mu, VNDims[mu]);
       }
     }
     
     _info=std::make_shared<LatticeInfo>(l_orig,l_dims,_g_info.GetNumSpins(), _g_info.GetNumColors(), _g_info.GetNodeInfo());
     
     // Init the data
     _cb_data=DataType("cb_data", _info->GetNumCBSites());
   }


   inline
     const LatticeInfo& GetInfo() const {
     return *(_info);
   }

   inline 
     const LatticeInfo& GetGlobalInfo() const {
     return _g_info;
   }


   inline
     IndexType GetCB() const {
     return _cb;
   }

   using VecType = SIMDComplex<typename BaseType<T>::Type, VN::VecLen>;
   using DataType = Kokkos::View<VecType*[4][3][3],GaugeLayout,MemorySpace>;


   KOKKOS_INLINE_FUNCTION
   const DataType& GetData() const {
     return _cb_data;
   }
  
   KOKKOS_INLINE_FUNCTION 
   DataType& GetData() {
     return _cb_data;
   }

   
 private:
   DataType _cb_data;
   const LatticeInfo& _g_info;
   std::shared_ptr<LatticeInfo> _info;
   const IndexType _cb;
 };

 template<typename T, typename VN>
   class KokkosFineVGaugeField {
 private:
   const LatticeInfo& _info;
   KokkosCBFineVGaugeField<T,VN>  _gauge_data_even;
   KokkosCBFineVGaugeField<T,VN>  _gauge_data_odd;
 public:
 KokkosFineVGaugeField(const LatticeInfo& info) :  _info(info), _gauge_data_even(info,EVEN), _gauge_data_odd(info,ODD) {
		}

   const KokkosCBFineVGaugeField<T,VN>& operator()(IndexType cb) const
     {
       return  (cb == EVEN) ? _gauge_data_even : _gauge_data_odd;
       //return *(_gauge_data[cb]);
     }
   
   KokkosCBFineVGaugeField<T,VN>& operator()(IndexType cb) {
     return (cb == EVEN) ? _gauge_data_even : _gauge_data_odd;
     //return *(_gauge_data[cb]);
   }
 };

 // Double copied gauge field.






 template<typename T, typename VN>
 class KokkosCBFineVGaugeFieldDoubleCopy {
 public:
 KokkosCBFineVGaugeFieldDoubleCopy(const LatticeInfo& info, IndexType cb)
   : _g_info(info), _cb(cb) {

     if( _g_info.GetNumColors() != 3 ) {
       MasterLog(ERROR, "KokkosCBFineSpinor has to have 3 colors in info. Info has %d", _g_info.GetNumColors());
     }
    
     IndexArray l_orig = _g_info.GetLatticeOrigin();
     IndexArray l_dims = _g_info.GetLatticeDimensions();
     IndexArray VNDims = { VN::Dim0, VN::Dim1, VN::Dim2, VN::Dim3};
     for(int mu=0; mu < 4; ++mu ) {
       if( l_dims[mu] % VNDims[mu] == 0 ) { 
	 l_orig[mu] /= VNDims[mu];
	 l_dims[mu] /= VNDims[mu];
       }
       else{
	 MasterLog(ERROR, "Local dimension %d (=%d) not divisible by VNode::dims[%d]=%d",
		   mu, l_dims[mu], mu, VNDims[mu]);
       }
     }
     
     _info=std::make_shared<LatticeInfo>(l_orig,l_dims,_g_info.GetNumSpins(), _g_info.GetNumColors(), _g_info.GetNodeInfo());
     
     // Init the data
     _cb_data=DataType("cb_data", _info->GetNumCBSites());

     MasterLog(INFO, "Exiting Constructor");
   }


   inline
     const LatticeInfo& GetInfo() const {
     return *(_info);
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
   using VecType = SIMDComplex<typename BaseType<T>::Type, VN::VecLen>;
   using DataType = Kokkos::View<VecType*[8][3][3],GaugeLayout,MemorySpace>;

   KOKKOS_INLINE_FUNCTION
   const DataType& GetData() const {
     return _cb_data;
   }
  
   KOKKOS_INLINE_FUNCTION 
   DataType& GetData() {
     return _cb_data;
   }

   void import(const KokkosCBFineVGaugeField<T,VN>& src_cb,
	       const KokkosCBFineVGaugeField<T,VN>& src_othercb)
   {
     using InputType = typename KokkosCBFineVGaugeField<T,VN>::DataType;

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

     IndexArray cb_latdims = _info->GetCBLatticeDimensions();
     SiteTable<VN> neigh_table(cb_latdims[0],cb_latdims[1],cb_latdims[2],
			       cb_latdims[3]);
     // Iterate sites with MDRange -- no blocking for now.
     MDPolicy policy({0,0,0,0},
		     {cb_latdims[0],cb_latdims[1],cb_latdims[2],cb_latdims[3]});
     
     const InputType cb_data_in = src_cb.GetData();
     const InputType othercb_data_in = src_othercb.GetData();
     //DataType l_cb_data = _cb_data; // Workaround
     int target_cb = _cb; // Lambd cannot access member _cb on host
			

     Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const int xcb,
						const int y,
						const int z,
						const int t) {	
			    int site = neigh_table.coords_to_idx(xcb,y,z,t);
			    int n_idx;
			    typename VN::MaskType mask;
			    
			    // T_minus
#if defined(MG_KOKKOS_USE_NEIGHBOR_TABLE)
			    neigh_table.NeighborTMinus(site,target_cb,n_idx,mask);
#else
			    neigh_table.NeighborTMinus(xcb,y,z,t,n_idx,mask);
#endif

			    for(int col=0; col < 3; ++col) {
			      for(int col2=0; col2 < 3; ++col2) { 
			        ComplexCopy(_cb_data(site,0,col,col2),VN::permute(mask, othercb_data_in(n_idx,3,col,col2)));
			      }
			    }

			    // Z_minus
#if defined(MG_KOKKOS_USE_NEIGHBOR_TABLE)
			    neigh_table.NeighborZMinus(site,target_cb,n_idx,mask);
#else
			    neigh_table.NeighborZMinus(xcb,y,z,t,n_idx,mask);
#endif
			    
			    for(int col=0; col < 3; ++col) {
			      for(int col2=0; col2 < 3; ++col2) { 
			        ComplexCopy(_cb_data(site,1,col,col2),VN::permute(mask,othercb_data_in(n_idx,2,col,col2)));
			      }
			    }

			    // Y_minus
#if defined(MG_KOKKOS_USE_NEIGHBOR_TABLE)
			    neigh_table.NeighborYMinus(site,target_cb,n_idx,mask);
#else
			    neigh_table.NeighborYMinus(xcb,y,z,t,n_idx,mask);
#endif
 
			    for(int col=0; col < 3; ++col) {
			      for(int col2=0; col2 < 3; ++col2) { 
			        ComplexCopy(_cb_data(site,2,col,col2),VN::permute(mask,othercb_data_in(n_idx,1,col,col2)));
			      }
			    }

			    // X_minus
#if defined(MG_KOKKOS_USE_NEIGHBOR_TABLE)
			    neigh_table.NeighborXMinus(site,target_cb,n_idx,mask);
#else
			    neigh_table.NeighborXMinus(xcb,y,z,t,target_cb,n_idx,mask);
#endif

			    
			    for(int col=0; col < 3; ++col) {
			      for(int col2=0; col2 < 3; ++col2) { 
			        ComplexCopy(_cb_data(site,3,col,col2),VN::permute(mask, othercb_data_in(n_idx,0,col,col2)));
			      }
			    }

			    for(int col=0; col < 3; ++col) {
			      for(int col2=0; col2 < 3; ++col2) { 
			        ComplexCopy(_cb_data(site,4,col,col2),cb_data_in(site,0,col,col2));
			      }
			    }

			    // Y_Plus
			    for(int col=0; col < 3; ++col) {
			      for(int col2=0; col2 < 3; ++col2) { 
			        ComplexCopy(_cb_data(site,5,col,col2),cb_data_in(site,1,col,col2));
			      }
			    }

			    // Z_Plus
			    for(int col=0; col < 3; ++col) {
			      for(int col2=0; col2 < 3; ++col2) { 
			        ComplexCopy(_cb_data(site,6,col,col2),cb_data_in(site,2,col,col2));
			      }
			    }

			    // T_Plus
			    for(int col=0; col < 3; ++col) {
			      for(int col2=0; col2 < 3; ++col2) { 
			        ComplexCopy(_cb_data(site,7,col,col2),cb_data_in(site,3,col,col2));
			      }
			    }
			    
			  });



   }
   
 private:
   DataType _cb_data;
   const LatticeInfo& _g_info;
   std::shared_ptr<LatticeInfo> _info;
   const IndexType _cb;
 };

 template<typename T, typename VN>
   void import(KokkosCBFineVGaugeFieldDoubleCopy<T,VN>& target,
	       const KokkosCBFineVGaugeField<T,VN>& src_cb,
	       const KokkosCBFineVGaugeField<T,VN>& src_othercb)
 {
     using InputType = typename KokkosCBFineVGaugeField<T,VN>::DataType;

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

     SiteTable<VN> neigh_table(cb_latdims[0],cb_latdims[1],cb_latdims[2],
			       cb_latdims[3]);

#ifdef MG_KOKKOS_USE_MDRANGE
     MDPolicy policy({0,0,0,0},
		     {cb_latdims[0],cb_latdims[1],cb_latdims[2],cb_latdims[3]},
		     {1,1,1,1});
#endif

     auto cb_data_out = target.GetData();
     auto cb_data_in = src_cb.GetData();
     auto othercb_data_in = src_othercb.GetData();

     int target_cb = target.GetCB(); // Lambd cannot access member _cb on host
     int num_cbsites = info.GetNumCBSites();	
#ifdef MG_KOKKOS_USE_MDRANGE
     Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const int xcb,
						const int y,
						const int z,
						const int t) {	
			    int site = neigh_table.coords_to_idx(xcb,y,z,t);
#else
     Kokkos::parallel_for(num_cbsites, KOKKOS_LAMBDA(const int site) { 
			    int xcb, y, z, t;
			    neigh_table.idx_to_coords(site,xcb,y,z,t);
#endif

			    int n_idx;
			    typename VN::MaskType mask;
			     
			    // T_minus
#if defined(MG_KOKKOS_USE_NEIGHBOR_TABLE)
			    neigh_table.NeighborTMinus(site,target_cb,n_idx,mask);
#else
			    neigh_table.NeighborTMinus(xcb,y,z,t,n_idx,mask);
#endif

			    for(int col=0; col < 3; ++col) {
			      for(int col2=0; col2 < 3; ++col2) { 
			        ComplexCopy(cb_data_out(site,0,col,col2),VN::permute(mask, othercb_data_in(n_idx,3,col,col2)));
			        //cb_data_out(site,0,col,col2) = VN::permute(mask, othercb_data_in(n_idx,3,col,col2));
			      }
			    }

			    // Z_minus
#if defined(MG_KOKKOS_USE_NEIGHBOR_TABLE)
			    neigh_table.NeighborZMinus(site,target_cb,n_idx,mask);
#else
			    neigh_table.NeighborZMinus(xcb,y,z,t,n_idx,mask);
#endif
			    
			    for(int col=0; col < 3; ++col) {
			      for(int col2=0; col2 < 3; ++col2) { 
				ComplexCopy(cb_data_out(site,1,col,col2),VN::permute(mask,othercb_data_in(n_idx,2,col,col2)));
				//cb_data_out(site,1,col,col2) = VN::permute(mask, othercb_data_in(n_idx,2,col,col2));
			      }
			    }

			    // Y_minus
#if defined(MG_KOKKOS_USE_NEIGHBOR_TABLE)
			    neigh_table.NeighborYMinus(site,target_cb,n_idx,mask);
#else
			    neigh_table.NeighborYMinus(xcb,y,z,t,n_idx,mask);
#endif
 
			    for(int col=0; col < 3; ++col) {
			      for(int col2=0; col2 < 3; ++col2) { 
			        ComplexCopy(cb_data_out(site,2,col,col2),VN::permute(mask,othercb_data_in(n_idx,1,col,col2)));
				//cb_data_out(site,2,col,col2) = VN::permute(mask, othercb_data_in(n_idx,1,col,col2));
			      }
			    }

			    // X_minus
#if defined(MG_KOKKOS_USE_NEIGHBOR_TABLE)
			    neigh_table.NeighborXMinus(site,target_cb,n_idx,mask);
#else
			    neigh_table.NeighborXMinus(xcb,y,z,t,target_cb,n_idx,mask);
#endif

			    
			    for(int col=0; col < 3; ++col) {
			      for(int col2=0; col2 < 3; ++col2) { 
			        ComplexCopy(cb_data_out(site,3,col,col2),VN::permute(mask, othercb_data_in(n_idx,0,col,col2)));
				//cb_data_out(site,3,col,col2) = VN::permute(mask, othercb_data_in(n_idx,0,col,col2));
			      }
			    }

			    for(int col=0; col < 3; ++col) {
			      for(int col2=0; col2 < 3; ++col2) { 
			        ComplexCopy(cb_data_out(site,4,col,col2),cb_data_in(site,0,col,col2));
				//cb_data_out(site,4,col,col2)=cb_data_in(site,0,col,col2);
			      }
			    }

			    // Y_Plus
			    for(int col=0; col < 3; ++col) {
			      for(int col2=0; col2 < 3; ++col2) { 
			        ComplexCopy(cb_data_out(site,5,col,col2),cb_data_in(site,1,col,col2));
				//cb_data_out(site,5,col,col2)=cb_data_in(site,1,col,col2);
			      }
			    }

			    // Z_Plus
			    for(int col=0; col < 3; ++col) {
			      for(int col2=0; col2 < 3; ++col2) { 
			        ComplexCopy(cb_data_out(site,6,col,col2),cb_data_in(site,2,col,col2));
				//cb_data_out(site,6,col,col2)=cb_data_in(site,2,col,col2);
			      }
			    }

			    // T_Plus
			    for(int col=0; col < 3; ++col) {
			      for(int col2=0; col2 < 3; ++col2) { 
			        ComplexCopy(cb_data_out(site,7,col,col2),cb_data_in(site,3,col,col2));
				//cb_data_out(site,7,col,col2)=cb_data_in(site,3,col,col2);
			      }
			    }
			    
			  });



   }


 template<typename T,typename VN>
   using VGaugeView = typename KokkosCBFineVGaugeFieldDoubleCopy<T,VN>::DataType;


}

#endif
