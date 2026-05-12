# Liquid 4.0.3 calls String#tainted? which was removed in Ruby 3.2.
# Patch String to restore the method as a no-op.
class String
  def tainted?
    false
  end unless method_defined?(:tainted?)
end
