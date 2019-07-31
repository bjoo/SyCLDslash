/*
 * kokkos_qdp_utils.h
 *
 *  Created on: May 23, 2017
 *      Author: bjoo
 */

#ifndef TEST_KOKKOS_KOKKOS_QDP_UTILS_H_
#define TEST_KOKKOS_KOKKOS_QDP_UTILS_H_

#include "qdp.h"
#include "kokkos_types.h"
#include <Kokkos_Core.hpp>

#include <utils/print_utils.h>
#include "kokkos_defaults.h"
#include "kokkos_vectype.h"
namespace MG
{

	// Single QDP++ Vector
	template<typename T, typename LF>
	void
	QDPLatticeFermionToKokkosCBSpinor(const LF& qdp_in,
			KokkosCBFineSpinor<MGComplex<T>,4>& kokkos_out)
	{
		auto cb = kokkos_out.GetCB();
		const QDP::Subset& sub = ( cb == EVEN ) ? QDP::rb[0] : QDP::rb[1];

		// Check conformance:
		int num_sites=static_cast<int>(kokkos_out.GetInfo().GetNumCBSites());

		if ( sub.numSiteTable() != num_sites ) {
			MasterLog(ERROR, "%s QDP++ Spinor has different number of sites per checkerboard than the KokkosCBFineSpinor",
					__FUNCTION__);
		}
		auto h_out = Kokkos::create_mirror_view( kokkos_out.GetData() );

		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::OpenMP>(0,num_sites),
				[=](int i) {
						for(int color=0; color < 3; ++color) {
							for(int spin=0; spin < 4; ++spin) {

							const int qdp_index = sub.siteTable()[i];

							h_out(i,color,spin)= MGComplex<T>(qdp_in.elem(qdp_index).elem(spin).elem(color).real(),
																	qdp_in.elem(qdp_index).elem(spin).elem(color).imag());

						} // spin
					} // color
			}// kokkos lambda
		);

		Kokkos::deep_copy(kokkos_out.GetData(), h_out);
	}

	// QDP N-vecor
	template<typename T, int N, typename LF>
	void
	QDPLatticeFermionToKokkosCBSpinor(const QDP::multi1d<LF>& qdp_in,
			KokkosCBFineSpinorVec<T,N>& kokkos_out)
	{
		if( qdp_in.size() != N)  {
			MasterLog(ERROR, "%s: multi1d<LF> array size (%d) is different from kokkos vector size (%d)",
					__FUNCTION__, qdp_in.size(), N );
		}

		auto cb = kokkos_out.GetCB();
		const QDP::Subset& sub = ( cb == EVEN ) ? QDP::rb[0] : QDP::rb[1];

		// Check conformance:
		int num_sites=static_cast<int>(kokkos_out.GetInfo().GetNumCBSites());

		if ( sub.numSiteTable() != num_sites ) {
			MasterLog(ERROR, "%s QDP++ Spinor has different number of sites per checkerboard than the KokkosCBFineSpinor",
					__FUNCTION__);
		}
		auto h_out = Kokkos::create_mirror_view( kokkos_out.GetData() );

		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::OpenMP>(0,num_sites),
				[=](int i) {
						for(int color=0; color < 3; ++color) {
							for(int spin=0; spin < 4; ++spin) {

							const int qdp_index = sub.siteTable()[i];

							for(int v=0; v < N; ++v ) {
								h_out(i,color,spin).set(v, MGComplex<T>(qdp_in[v].elem(qdp_index).elem(spin).elem(color).real(),
										qdp_in[v].elem(qdp_index).elem(spin).elem(color).imag()));
							}

						} // spin
					} // color
			}// kokkos lambda
		);

		Kokkos::deep_copy(kokkos_out.GetData(), h_out);
	}

	// Single QDP++ vector
	template<typename T, typename LF>
	void
	KokkosCBSpinorToQDPLatticeFermion(const KokkosCBFineSpinor<MGComplex<T>,4>& kokkos_in,
			LF& qdp_out) {

		auto cb = kokkos_in.GetCB();
		const QDP::Subset& sub = ( cb == EVEN ) ? QDP::rb[0] : QDP::rb[1];

		// Check conformance:
		int num_sites=static_cast<int>(kokkos_in.GetInfo().GetNumCBSites());

		if ( sub.numSiteTable() != num_sites ) {
			MasterLog(ERROR, "%s: QDP++ Spinor has different number of sites per checkerboard than the KokkosCBFineSpinor",
					__FUNCTION__);
		}

		auto h_in = Kokkos::create_mirror_view( kokkos_in.GetData() );
		Kokkos::deep_copy( h_in, kokkos_in.GetData() );

		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::OpenMP>(0,num_sites),
				[&](int i) {

						for(int color=0; color < 3; ++color) {
							for(int spin=0; spin < 4; ++spin) {
							const int qdp_index = sub.siteTable()[i];
							qdp_out.elem(qdp_index).elem(spin).elem(color).real() = h_in(i,color,spin).real();
							qdp_out.elem(qdp_index).elem(spin).elem(color).imag() = h_in(i,color,spin).imag();
						} // spin
					} // color
			}// kokkos lambda
		);


	}

	// QDP N-vector
	template<typename T, int N, typename LF>
	void
	KokkosCBSpinorToQDPLatticeFermion(const KokkosCBFineSpinorVec<T,N>& kokkos_in,
			QDP::multi1d<LF>& qdp_out) {

		if( qdp_out.size() != N)  {
			MasterLog(ERROR, "%s: multi1d<LF> array size (%d) is different from kokkos vector size (%d)",
						__FUNCTION__, qdp_out.size(), N );
		}
		auto cb = kokkos_in.GetCB();
		const QDP::Subset& sub = ( cb == EVEN ) ? QDP::rb[0] : QDP::rb[1];

		// Check conformance:
		int num_sites=static_cast<int>(kokkos_in.GetInfo().GetNumCBSites());

		if ( sub.numSiteTable() != num_sites ) {
			MasterLog(ERROR, "%s: QDP++ Spinor has different number of sites per checkerboard than the KokkosCBFineSpinor",
					__FUNCTION__);
		}

		auto h_in = Kokkos::create_mirror_view( kokkos_in.GetData() );
		Kokkos::deep_copy( h_in, kokkos_in.GetData() );

		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::OpenMP>(0,num_sites),
				[&](int i) {

						for(int color=0; color < 3; ++color) {
							for(int spin=0; spin < 4; ++spin) {
							const int qdp_index = sub.siteTable()[i];
							for(int v = 0; v < N; ++v) {
								qdp_out[v].elem(qdp_index).elem(spin).elem(color).real() = (h_in(i,color,spin)(v)).real();
								qdp_out[v].elem(qdp_index).elem(spin).elem(color).imag() = (h_in(i,color,spin)(v)).imag();
							}
						} // spin
					} // color
			}// kokkos lambda
		);


	}


	// Single QDP++ vector
	template<typename T, typename HF>
	void
	QDPLatticeHalfFermionToKokkosCBSpinor2(const HF& qdp_in,
			KokkosCBFineSpinor<MGComplex<T>,2>& kokkos_out)
	{
		auto cb = kokkos_out.GetCB();
		const QDP::Subset& sub = ( cb == EVEN ) ? QDP::rb[0] : QDP::rb[1];

		// Check conformance:
		int num_sites=static_cast<int>(kokkos_out.GetInfo().GetNumCBSites());

		if ( sub.numSiteTable() != num_sites ) {
			MasterLog(ERROR, "%s QDP++ Spinor has different number of sites per checkerboard than the KokkosCBFineSpinor",
					__FUNCTION__);
		}
		auto h_out = Kokkos::create_mirror_view( kokkos_out.GetData() );

		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::OpenMP>(0,num_sites),
				[=](int i) {
						for(int color=0; color < 3; ++color) {
							for(int spin=0; spin < 2; ++spin) {

							const int qdp_index = sub.siteTable()[i];

							h_out(i,color,spin) = MGComplex<T>(qdp_in.elem(qdp_index).elem(spin).elem(color).real(),
																	   qdp_in.elem(qdp_index).elem(spin).elem(color).imag());

						} // spin
					} // color
			}// kokkos lambda
		);

		Kokkos::deep_copy(kokkos_out.GetData(), h_out);
	}

	// N vector
	template<typename T, int N, typename HF>
	void
	QDPLatticeHalfFermionToKokkosCBSpinor2(const QDP::multi1d<HF>& qdp_in,
			KokkosCBFineHalfSpinorVec<T,N>& kokkos_out)
	{
		if( qdp_in.size() != N)  {
			MasterLog(ERROR, "%s: multi1d<LF> array size (%d) is different from kokkos vector size (%d)",
					__FUNCTION__, qdp_in.size(), N );
		}

		auto cb = kokkos_out.GetCB();
		const QDP::Subset& sub = ( cb == EVEN ) ? QDP::rb[0] : QDP::rb[1];

		// Check conformance:
		int num_sites=static_cast<int>(kokkos_out.GetInfo().GetNumCBSites());

		if ( sub.numSiteTable() != num_sites ) {
			MasterLog(ERROR, "%s QDP++ Spinor has different number of sites per checkerboard than the KokkosCBFineSpinor",
					__FUNCTION__);
		}
		auto h_out = Kokkos::create_mirror_view( kokkos_out.GetData() );

		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::OpenMP>(0,num_sites),
				[=](int i) {
						for(int color=0; color < 3; ++color) {
							for(int spin=0; spin < 2; ++spin) {

							const int qdp_index = sub.siteTable()[i];

							for(int v=0; v < N; ++v) {
								h_out(i,color,spin).set(v, MGComplex<T>(qdp_in[v].elem(qdp_index).elem(spin).elem(color).real(),
										qdp_in[v].elem(qdp_index).elem(spin).elem(color).imag()));
							}

						} // spin
					} // color
			}// kokkos lambda
		);

		Kokkos::deep_copy(kokkos_out.GetData(), h_out);
	}

	template<typename T, typename HF>
	void
	KokkosCBSpinor2ToQDPLatticeHalfFermion(const KokkosCBFineSpinor<MGComplex<T>,2>& kokkos_in,
			HF& qdp_out) {

		auto cb = kokkos_in.GetCB();
		const QDP::Subset& sub = ( cb == EVEN ) ? QDP::rb[0] : QDP::rb[1];

		// Check conformance:
		int num_sites=static_cast<int>(kokkos_in.GetInfo().GetNumCBSites());

		if ( sub.numSiteTable() != num_sites ) {
			MasterLog(ERROR, "%s: QDP++ Spinor has different number of sites per checkerboard than the KokkosCBFineSpinor",
					__FUNCTION__);
		}

		auto h_in = Kokkos::create_mirror_view( kokkos_in.GetData() );
		Kokkos::deep_copy( h_in, kokkos_in.GetData() );

		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::OpenMP>(0,num_sites),
				[&](int i) {
						for(int color=0; color < 3; ++color) {
							for(int spin=0; spin < 2; ++spin) {

							const int qdp_index = sub.siteTable()[i];
							qdp_out.elem(qdp_index).elem(spin).elem(color).real() = h_in(i,color,spin).real();
							qdp_out.elem(qdp_index).elem(spin).elem(color).imag() = h_in(i,color,spin).imag();
						} // spin
					} // color
			}// kokkos lambda
		);
	}

	template<typename T, int N, typename HF>
		void
		KokkosCBSpinor2ToQDPLatticeHalfFermion(const KokkosCBFineHalfSpinorVec<T,N>& kokkos_in,
				QDP::multi1d<HF>& qdp_out) {

			if( qdp_out.size() != N)  {
					MasterLog(ERROR, "%s: multi1d<LF> array size (%d) is different from kokkos vector size (%d)",
							__FUNCTION__, qdp_out.size(), N );
				}

			auto cb = kokkos_in.GetCB();
			const QDP::Subset& sub = ( cb == EVEN ) ? QDP::rb[0] : QDP::rb[1];

			// Check conformance:
			int num_sites=static_cast<int>(kokkos_in.GetInfo().GetNumCBSites());

			if ( sub.numSiteTable() != num_sites ) {
				MasterLog(ERROR, "%s: QDP++ Spinor has different number of sites per checkerboard than the KokkosCBFineSpinor",
						__FUNCTION__);
			}

			auto h_in = Kokkos::create_mirror_view( kokkos_in.GetData() );
			Kokkos::deep_copy( h_in, kokkos_in.GetData() );

			Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::OpenMP>(0,num_sites),
					[&](int i) {
							for(int color=0; color < 3; ++color) {
								for(int spin=0; spin < 2; ++spin) {

								const int qdp_index = sub.siteTable()[i];
								for(int v=0; v < N; ++v ) {
									qdp_out[v].elem(qdp_index).elem(spin).elem(color).real() = (h_in(i,color,spin)(v)).real();
									qdp_out[v].elem(qdp_index).elem(spin).elem(color).imag() = (h_in(i,color,spin)(v)).imag();
								}
							} // spin
						} // color
				}// kokkos lambda
			);

	}

	template<typename T, typename GF>
	void
	QDPGaugeFieldToKokkosCBGaugeField(const GF& qdp_in,
			KokkosCBFineGaugeField<MGComplex<T>>& kokkos_out)
	{
		auto cb = kokkos_out.GetCB();
		const QDP::Subset& sub = ( cb == EVEN ) ? QDP::rb[0] : QDP::rb[1];

		// Check conformance:
		int num_sites=static_cast<int>(kokkos_out.GetInfo().GetNumCBSites());

		if ( sub.numSiteTable() != num_sites ) {
			MasterLog(ERROR, "%s QDP++ Gauge has different number of sites per checkerboard than the KokkosCBFineSpinor",
					__FUNCTION__);
		}
		if ( qdp_in.size() != n_dim ) {
			MasterLog(ERROR, "%s QDP++ Gauge has wrong number of dimensions (%d instead of %d)",
						__FUNCTION__, qdp_in.size(), n_dim);
		}

		auto h_out = Kokkos::create_mirror_view( kokkos_out.GetData() );

		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::OpenMP>(0,num_sites),
				[=](int i) {
					for(int mu=0; mu< 4; ++mu) {
						for(int color=0; color < 3; ++color) {
							for(int color2=0; color2 < 3; ++color2) {
								const int qdp_index = sub.siteTable()[i];

								h_out(i,mu,color,color2)=MGComplex<T>( (qdp_in[mu]).elem(qdp_index).elem().elem(color,color2).real(),
																			 (qdp_in[mu]).elem(qdp_index).elem().elem(color,color2).imag());
							} //color2
						} // color
					} // mu

			});
		Kokkos::deep_copy(kokkos_out.GetData(), h_out);
	}




	template<typename T, typename GF>
	void
	KokkosCBGaugeFieldToQDPGaugeField(const KokkosCBFineGaugeField<MGComplex<T>>& kokkos_in,
			GF& qdp_out)
	{
		auto cb = kokkos_in.GetCB();

		const QDP::Subset& sub = ( cb == EVEN ) ? QDP::rb[0] : QDP::rb[1];

		// Check conformance:
		int num_sites=static_cast<int>(kokkos_in.GetInfo().GetNumCBSites());

		if ( sub.numSiteTable() != num_sites ) {
			MasterLog(ERROR, "%s QDP++ Gauge has different number of sites per checkerboard than the KokkosCBFineSpinor",
					__FUNCTION__);
		}
		if ( qdp_out.size() != n_dim ) {
			qdp_out.resize(n_dim);
		}

		// Creating Host Mirror
		auto h_in = Kokkos::create_mirror_view( kokkos_in.GetData() );
		Kokkos::deep_copy(h_in, kokkos_in.GetData());

		for(int mu=0; mu < 4; ++mu ) {
		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::OpenMP>(0,num_sites),
				[&](int i) {
			       for(int color=0; color < 3; ++color) {
			    	   for(int color2=0; color2 < 3; ++color2) {
			    		   const int qdp_index = sub.siteTable()[i];


			    		   (qdp_out[mu]).elem(qdp_index).elem().elem(color,color2).real()=
			    				   h_in(i,mu,color,color2).real();
			    		   (qdp_out[mu]).elem(qdp_index).elem().elem(color,color2).imag()=
									h_in(i,mu,color,color2).imag();
						} //color2
					} // color

			   }// kokkos lambda
		);

		} // mu

	}

	template<typename T, typename GF>
	void
	QDPGaugeFieldToKokkosGaugeField(const GF& qdp_in,
									KokkosFineGaugeField<T>& kokkos_out)
	{
		QDPGaugeFieldToKokkosCBGaugeField( qdp_in, kokkos_out(EVEN));
		QDPGaugeFieldToKokkosCBGaugeField( qdp_in, kokkos_out(ODD));
	}

	template<typename T, typename GF>
	void
	KokkosGaugeFieldToQDPGaugeField(const KokkosFineGaugeField<T>& kokkos_in,
									GF& qdp_out)
	{
		KokkosCBGaugeFieldToQDPGaugeField( kokkos_in(EVEN),qdp_out);
		KokkosCBGaugeFieldToQDPGaugeField( kokkos_in(ODD), qdp_out);
	}

}




#endif /* TEST_KOKKOS_KOKKOS_QDP_UTILS_H_ */
