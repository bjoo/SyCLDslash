/*
 * sycl_view.h
 *
 *  Created on: Jul 29, 2019
 *      Author: bjoo
 */

#pragma once

#include <cstddef>
#include <CL/sycl.hpp>

namespace MG {
// Some Constexpr functions to calculate at compile time
// the sizes in the body.
class BodySize {
public:
	// Dim = 1
	static
	inline constexpr
	size_t bodySize( std::array<size_t,1> dims )  { return dims[0]; }

	// Dim = 2
	static
	inline constexpr
	size_t bodySize( std::array<size_t,2> dims )  {
		return dims[0]*dims[1];
	}

	// Dim = 3
	static
	inline constexpr
	size_t bodySize( std::array<size_t,3> dims ) {
		return dims[0]*dims[1]*dims[2];
	}

	// Dim = 4
	static
	inline constexpr
	size_t bodySize( std::array<size_t,4> dims )  {
		return dims[0]*dims[1]*dims[2]*dims[3];
	}

};

// LayoutLeft class
//
//  Provides indexing:  index()  (up to 4 dims)
//  convert idex to coordinates: coords()   (up to 4 dims)
//
//  assuming leftmost index runs fastest
class LayoutLeft {
public:
	// Dim=1
	static
	inline constexpr
	size_t index( const std::array<size_t,1>& idx,
				  const std::array<size_t,1>& dims ) { return idx[0]; }

	static
	inline
	std::array<size_t,1> coords(size_t idx, const std::array<size_t,1>& dims)
	{
		std::array<size_t,1> result={0};
		result[0] = idx;
		return result;
	}

	// Dim=2
	static
	inline constexpr
	size_t index( const std::array<size_t,2>& idx,
				  const std::array<size_t,2>& dims )  {
		return idx[0] + dims[0]*idx[1];
	}

	static
	inline
	std::array<size_t,2> coords(size_t idx, const std::array<size_t,2>& dims) {
		std::array<size_t,2> result={0,0};

		result[0] = idx % dims[0];
		result[1] = idx / dims[0];

		return result;
	}


	// Dim=3
	static
	inline constexpr
	size_t index( std::array<size_t,3> idx, std::array<size_t,3> dims )  {
		return idx[0] + dims[0]*(idx[1] + dims[1]*idx[2]);
	}

	static
	inline
	std::array<size_t,3> coords(size_t idx, const std::array<size_t,3>& dims) {
		std::array<size_t,3> result={0,0,0};

		result[0] = idx % dims[0];
		size_t tmp = idx / dims[0];

		result[1] = tmp % dims[1];

		result[2] = tmp / dims[1];

		return result;

	}

	// Dim=4
	static
	inline constexpr
	size_t index( std::array<size_t,4> idx, std::array<size_t,4> dims ) {
		return idx[0] + dims[0]*(idx[1] + dims[1]*(idx[2] + dims[2]*idx[3]));
	}

	static
	inline
	std::array<size_t,4> coords(size_t idx, const std::array<size_t,4>& dims) {

		std::array<size_t,4> result={0,0,0,0};

		result[0] = idx % dims[0];
		size_t tmp = idx / dims[0];     // discard remainder

		result[1] = tmp % dims[1];
		size_t tmp2 = tmp / dims[1];

		result[2] = tmp2 % dims[2];
		result[3] = tmp2 / dims[2];
		return result;

	}

};

// LayoutLeft class
//
//  Provides indexing:  index()  (up to 4 dims)
//  convert idex to coordinates: coords()   (up to 4 dims)
//
//  assuming rightmost index runs fastest

class LayoutRight {
public:
	// Dim=1
	static
	inline constexpr
	size_t index( std::array<size_t,1> idx, std::array<size_t,1> dims ) { return idx[0]; }

	static
	inline
	std::array<size_t,1> coords(size_t idx, const std::array<size_t,1>& dims) {
		std::array<size_t,1> result={0};
		result[0] = idx;
		return result;
	}

	// Dim=2
	static
	inline constexpr
	size_t index( std::array<size_t,2> idx, std::array<size_t,2> dims )  {
		return idx[1] + dims[1]*idx[0];
	}

	static
	inline
	std::array<size_t,2> coords(size_t idx, const std::array<size_t,2>& dims) {
		std::array<size_t,2> result={0,0};

		result[1] = idx % dims[1];
		result[0] = idx / dims[1];
		return result;
	}

	// Dim=3
	static
	inline constexpr
	size_t index( std::array<size_t,3> idx, std::array<size_t,3> dims )  {
		return idx[2] + dims[2]*(idx[1] + dims[1]*idx[0]);
	}


	static
	inline
	std::array<size_t,3> coords(size_t idx, const std::array<size_t,3>& dims) {
		std::array<size_t,3> result={0,0,0};

		result[2] = idx % dims[2];
		size_t tmp = idx / dims[2];

		result[1] = tmp % dims[1];
		result[0] = tmp / dims[1];

		return result;
	}

	// Dim=4
	static
	inline constexpr
	size_t index( std::array<size_t,4> idx, std::array<size_t,4> dims ) {
		return idx[3] + dims[3]*(idx[2] + dims[2]*(idx[1] + dims[1]*idx[0]));
	}

	static
	inline
	std::array<size_t,4> coords(size_t idx, const std::array<size_t,4>& dims) {
		std::array<size_t,4> result={0,0,0,0};

		result[3] = idx % dims[3];

		size_t tmp = idx / dims[3];
		result[2] = tmp % dims[2];

		size_t tmp2 = tmp / dims[2];
		result[1] = tmp2 % dims[1];

		result[0]  = tmp2 / dims[1];
		return result;
	}
};


// View accessor class. Instantiated with an accessor returned by get access.
//
template<typename T, std::size_t Ndim, typename Layout,
		cl::sycl::access::mode accessMode,
		cl::sycl::access::target accessTarget=cl::sycl::access::target::global_buffer>
class ViewAccessor {
public:
	ViewAccessor(const std::array<std::size_t,Ndim> dims,
				 cl::sycl::accessor<T,1,accessMode,accessTarget> accessor) : _dims(dims),_accessor(accessor) {}


	// The sycl spec has accessor functions for 1 dimensional buffers:
 	//     T &operato[] const -- if accessMode == write or read_write or discard_write
	// but T operator[] cost -- if accessMode == read
	//
	// annoyingly I can't overeload on return type, so I need to do a class specialization
	// for read-only views.

	// Dim == 1
	inline
	T& operator()(size_t i0) const {
		return _accessor[ Layout::index({i0}, _dims) ];
	}

	// Dim == 2
	inline
	 T& operator()(size_t i0, size_t i1) const {
		return _accessor[ Layout::index({i0, i1}, _dims) ];
	}

	// Dim == 3
	inline
	T& operator()(size_t i0, size_t i1, size_t i2) const  {
		return _accessor[ Layout::index({i0, i1, i2}, _dims) ];
	}


	T& operator()(size_t i0, size_t i1, size_t i2, size_t i3) const {
		return _accessor[ Layout::index({i0, i1, i2,i3}, _dims) ];
	}




	std::array<std::size_t,Ndim> _dims;
	cl::sycl::accessor<T,1,accessMode,accessTarget> _accessor;
};


// Class specialization of ViewAccessor for read-only views.
//
template<typename T, std::size_t Ndim, typename Layout,
		cl::sycl::access::target accessTarget>
class ViewAccessor<T,Ndim,Layout,cl::sycl::access::mode::read,accessTarget> {
public:
	ViewAccessor(const std::array<std::size_t,Ndim> dims,
				 cl::sycl::accessor<T,1,cl::sycl::access::mode::read,accessTarget> accessor) : _dims(dims),_accessor(accessor) {}

	// Dim == 1
	inline
	T operator()(size_t i0) const {
		return _accessor[ Layout::index({i0}, _dims) ];
	}

	// Dim == 2
	inline
	 T operator()(size_t i0, size_t i1) const {
		return _accessor[ Layout::index({i0, i1}, _dims) ];
	}

	// Dim == 3
	inline
	T operator()(size_t i0, size_t i1, size_t i2) const  {
		return _accessor[ Layout::index({i0, i1, i2}, _dims) ];
	}


	T operator()(size_t i0, size_t i1, size_t i2, size_t i3) const {
		return _accessor[ Layout::index({i0, i1, i2,i3}, _dims) ];
	}


	std::array<std::size_t,Ndim> _dims;
	cl::sycl::accessor<T,1,cl::sycl::access::mode::read,accessTarget> _accessor;
};



// The Main View type
//
// It allocates a buffer on instantiation. One can give it the buffer allocator.
//
// To access the data one can request ViewAccessors in a similar way to regular SyCL buffers
// using a getAccess method. Underneath the hood the view will get the accessor to the buffer and
// bury it inside ViewAccessor along with the layout for indexing.
//
// Everything is RAII
//
// There is a getAccess() call with a Command Group Handler for use inside Queues
// There is also a getAccess() method without a CGH for use on the host.
template<typename T, std::size_t Ndim, typename Layout, typename AllocatorT = cl::sycl::buffer_allocator>
class View {
public:

	View() : _inited(false), _name(""), _dims({0}), _buf(0) {}
	View(const std::string name, std::array<std::size_t, Ndim> dims) : _inited(true), _name(name), _dims(dims), _buf(BodySize::bodySize(_dims)) {}


	template<cl::sycl::access::mode accessMode>
	ViewAccessor<T,Ndim,Layout,accessMode,cl::sycl::access::target::host_buffer>  get_access()  {
		return ViewAccessor<T,Ndim,Layout,accessMode,cl::sycl::access::target::host_buffer>(_dims,_buf.template get_access<accessMode>());
	}

	template<cl::sycl::access::mode accessMode>
	ViewAccessor<T,Ndim,Layout,accessMode,cl::sycl::access::target::global_buffer>  get_access(cl::sycl::handler& cgh)  {
		return ViewAccessor<T,Ndim,Layout, accessMode,cl::sycl::access::target::global_buffer>(_dims,_buf.template get_access<accessMode>(cgh));
	}

	const std::string& getName() const { return _name; }
	const std::array<std::size_t,Ndim> getDims() const { return _dims; }
	const size_t getNumDims() const { return _dims.size(); }

	bool _inited;
	std::string _name;
	std::array<std::size_t,Ndim> _dims;
	cl::sycl::buffer<T,1,AllocatorT> _buf;
};

} // Namespace
