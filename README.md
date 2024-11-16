# 📅 Smart Calendar Project

## 🎯 What We Want to Do
Hey there! 👋 We're building a super smart calendar that will make scheduling tasks a breeze! 🌟 Our goal is to create a calendar that:

- [x] Automatically schedules tasks for you 📝
- [x] Understands your preferences and constraints 🤔
- [x] Works seamlessly with Google Calendar 📆
- [x] Makes the best use of your time ⏰

## 🎉 What We've Achieved So Far
### ✅ Core Features
1. **Automatic Task Scheduling**
   - [x] Just tell it the task name, how long it will take, and when it's due
   - [x] It knows different types of tasks (study, coding, meetings, and more!)
   - [x] It handles tasks with different priorities
   - [x] It finds the best time slots for you

2. **Integrations**
   - [x] Works perfectly with Google Calendar
   - [x] Uses Groq's awesome AI to make smart scheduling decisions
   - [x] Remembers your preferences and previous schedules

3. **Rules & Preferences**
   - [x] You can tell it your class schedules, sleep times, and when you're busy
   - [x] It learns what times you prefer for different types of tasks
   - [x] It reads your rules from a simple configuration file

### 🧪 Testing and Validation
- [x] We've tested it with all sorts of scenarios and edge cases
- [x] It works great with course registration, project demos, online exams, emergency meetings, research paper reviews, and client meetings
- [x] It handles timezones and durations like a champ

### 🛠️ Technical Stuff
- [x] The code is organized into neat little modules
- [x] It's all packaged up nicely with Docker for easy setup
- [x] It's timezone-aware and formats times correctly

## 🚀 What's Next
1. Support for even more specific time preferences
2. Better handling of last-minute tasks
3. Customizable breaks between tasks
4. Ability to split big tasks into smaller chunks
5. Support for more calendar types

## 🎬 Getting Started
1. Clone this repository
2. Build the Docker image using the Dockerfile
3. Run the container with the right environment variables and volume mounts

```bash
# Build Docker image
docker build -t smart-calendar .

# Run with environment variables
docker run -it --rm \
  -v "$(pwd)/config:/app/config" \
  -v "$(pwd)/cache:/app/cache" \
  -e GROQ_API_KEY="your_key_here" \
  smart-calendar
```

For more details about how everything works under the hood, check out our project docs! 📚
