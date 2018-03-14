#import Base.start, Base.next, Base.done
#type Combinations
  #from::Int64
  #choose::Int64
  #total::Int64
  #Combinations(from, choose) = new(from, choose, binomial(from, choose))
#end
#typealias CombinationState Tuple{Array{Int64,1},Int64}
#
#start(x::Combinations) = (collect(1:x.choose),1)
#done(x::Combinations, s::CombinationState) = s[2] >= x.total
#
#inc(c, i) = [c[1:i-1]; (c[i]+1):(c[i]+1+length(c)-i)]
##
#function next_i(x::Combinations, c, i)
  #if c[i] < x.from - (i - x.choose)
    #inc(c, i)
  #else
    #next_i(x, c, i-1)
  #end
#end
#
#function next(x::Combinations, s::CombinationState)
  #ith::Array{Int64,1} = next_i(x, s[1], x.choose)
  #(ith, (ith, (s[2]+1)::Int64))
#end
