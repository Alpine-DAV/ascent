#ifndef DRAY_YAML_WRITER_HPP
#define DRAY_YAML_WRITER_HPP

#include <sstream>
#include <stdexcept>
#include <stack>

class YamlWriter
{
  struct Block
  {
    int m_indent;
    bool m_is_list;
    bool m_at_list_item_start;

    Block(int indent)
      : m_indent(indent), m_is_list(false), m_at_list_item_start(false)
    {  }
  };

  std::stringstream m_output_stream;
  std::stack<Block> m_block_stack;
  bool m_at_block_start;

  Block &m_current_block()
  {
    return this->m_block_stack.top();
  }

  const Block &m_current_block() const
  {
    return this->m_block_stack.top();
  }

  void write_indent()
  {
    int indent = this->m_current_block().m_indent;
    bool listStart = this->m_current_block().m_at_list_item_start;

    if (listStart)
    {
      --indent;
    }

    for (int i = 0; i < indent; ++i)
    {
      this->m_output_stream << "  ";
    }

    if (listStart)
    {
      this->m_output_stream << "- ";
      this->m_current_block().m_at_list_item_start = false;
    }
  }

public:
  YamlWriter()
    : m_at_block_start(true)
  {
    this->m_block_stack.push(Block(0));
  }

  ~YamlWriter()
  {
    if (this->m_block_stack.size() != 1)
    {
      //throw std::runtime_error(
      //      "YamlWriter destroyed before last block complete.");
    }
  }
  
  std::stringstream& get_stream()
  {
    return m_output_stream;
  }
  /// Starts a block underneath a dictionary item. The key for the block is
  /// given, and the contents of the block, which can be a list or dictionary
  /// or list of dictionaries and can contain sub-blocks, is created by calling
  /// further methods of this class.
  ///
  /// A block started with \c StartBlock _must_ be ended with \c EndBlock.
  ///
  void start_block(const std::string &key)
  {
    this->write_indent();
    this->m_output_stream << key << ":" << std::endl;

    int indent = this->m_current_block().m_indent;
    this->m_block_stack.push(Block(indent+1));
    this->m_at_block_start = true;
  }

  /// Finishes a block.
  ///
  void end_block()
  {
    this->m_block_stack.pop();
    this->m_at_block_start = false;

    if (this->m_block_stack.empty())
    {
      throw std::runtime_error("Ended a block that was never started.");
    }
  }

  /// Start an item in a list. Can be a dictionary item.
  ///
  void start_list_item()
  {
    Block &block = this->m_current_block();
    if (block.m_is_list)
    {
      if (!block.m_at_list_item_start)
      {
        block.m_at_list_item_start = true;
      }
      else
      {
        // Ignore empty list items
      }
    }
    else if (this->m_at_block_start)
    {
      // Starting a list.
      block.m_is_list = true;
      block.m_at_list_item_start = true;
      ++block.m_indent;
    }
    else
    {
      throw std::runtime_error(
            "Tried to start a list in the middle of a yaml block.");
    }
  }

  /// Add a list item that is just a single value.
  ///
  void add_value(const std::string &value)
  {
    this->start_list_item();
    this->write_indent();
    this->m_output_stream << value << std::endl;
    this->m_at_block_start = false;
  }

  /// Add a key/value pair for a dictionary entry.
  ///
  template<typename T>
  void add_entry(const std::string &key, const T &value)
  {
    this->write_indent();
    this->m_output_stream << key << ": " << value << std::endl;
    this->m_at_block_start = false;
  }
};

#endif //_YamlWriter_h
