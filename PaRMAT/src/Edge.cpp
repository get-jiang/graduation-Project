#include "Edge.hpp"
#include <random>

// Global weight range variables (default: 10 to 1000)
int g_minWeight = 10;
int g_maxWeight = 1000;

Edge::Edge(EdgeIndexType rec_src, EdgeIndexType rec_dst):
		src(rec_src), dst(rec_dst)
{}

Edge::Edge(const Edge &cSource) {
	src = cSource.src;
	dst = cSource.dst;
}

Edge& Edge::operator= (const Edge &cSource) {
	src = cSource.src;
	dst = cSource.dst;
	return *this;
}

bool Edge::selfEdge(){
	return ( src == dst );
}

bool operator< (const Edge& cR1, const Edge& cR2) {
	if( cR1.src < cR2.src )
		return true;
	else if( cR1.src > cR2.src )
		return false;
	else if( cR1.dst < cR2.dst )
		return true;
	else
		return false;
}

bool operator== (const Edge& cR1, const Edge& cR2) {
	return ( cR1.dst == cR2.dst && cR1.src == cR2.src);
}

std::ostream& operator<< (std::ostream &out, Edge &cEdge) {
	thread_local std::mt19937 gen(std::random_device{}());
	std::uniform_int_distribution<int> dist(g_minWeight, g_maxWeight);
	out << cEdge.src + 1 << "\t" << cEdge.dst + 1 << "\t" << dist(gen) << "\n";
	return out;
}
