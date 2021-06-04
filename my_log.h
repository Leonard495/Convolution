#ifndef MY_LOG_H
#define MY_LOG_H
#include <chrono>
#include <string>
#undef MYLOG_ENABLED
#define MYMSG_ENABLED

#ifdef MYLOG_ENABLED
#	define LOG_INFO(fmt, args...)	printf(fmt, ##args)
#else
#	define LOG_INFO(fmt, args...)
#endif

#ifdef MYMSG_ENABLED
#	define LOG_MSG(fmt, args...)	printf(fmt, ##args)
#	define MSG(fmt, args...)																			\
{																										\
	std::string f_str = std::string(__FILE__) + ", " + std::to_string(__LINE__) + ": " + fmt + "\n";	\
	printf(f_str.c_str(), ##args);																		\
}
#else
#	define LOG_MSG(fmt, args...)
#endif


#define LOG_ERROR(fmt, args...)	printf(fmt, ##args)

#define LOG_FUNCTION() 						\
	{										\
											\
		LOG_MSG("%s, %d, %s entry point\n", __FILE__, __LINE__, __func__);	\
	}

class Tracker {
	std::chrono::steady_clock::time_point start;
	std::chrono::steady_clock::time_point finish;
	std::string text;
	public:
	Tracker(const char* text_) :
		text(std::string(text_))
	{
		start = std::chrono::steady_clock::now();
	};
	~Tracker(void) {
		finish = std::chrono::steady_clock::now();
		unsigned int interval = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count();
		LOG_MSG("##### [Time tracking] ##### %s, interval = %d us\n", text.c_str(), interval);
	}
};

#endif
