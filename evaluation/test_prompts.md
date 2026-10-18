# Evaluation prompts — Shiba Coder 27B Max

Ten prompts used for the A/B evaluation against the unmodified base model
(`prism-ml/Bonsai-27B-unpacked`). Prompts were chosen to probe the behaviours
the adapter was trained on: **code generation**, **debugging**,
**refactoring**, **feature implementation**, and **algorithmic reasoning**.

They are deliberately **not** drawn from the training data.

## 1. API endpoint (FastAPI)

> Create a Python FastAPI endpoint for user registration. Use Pydantic
> validation, proper typing, useful HTTP errors, and separate validation
> logic from persistence logic.

## 2. Bug diagnosis

> This Python function is supposed to return the second-largest element in a
> list but returns the wrong value for some inputs. Find the bug, explain it,
> and provide a fixed version with tests.
>
> ```python
> def second_largest(xs):
>     xs = list(set(xs))
>     xs.sort()
>     return xs[-2]
> ```

## 3. Refactor to SOLID

> Refactor the following class to follow the Single Responsibility Principle.
> Split it into at least two classes and explain the trade-off.
>
> ```python
> class OrderService:
>     def __init__(self, db): self.db = db
>     def create_order(self, data):
>         if not data.get("items"): raise ValueError("empty")
>         self.db.save(data)
>         self.send_email(data["customer"], "Order created")
>     def send_email(self, to, body): ...  # smtp logic
> ```

## 4. Algorithm (complexity-aware)

> Implement a function that returns the length of the longest increasing
> subsequence of an integer list. Include a complexity analysis and at least
> three test cases.

## 5. SQL + query optimization

> Given tables `users(id, email)` and `orders(id, user_id, amount, created_at)`,
> write a SQL query that returns, for each user, their total spend in the last
> 30 days. Then explain how you would index it for performance.

## 6. Type-safe data layer (Python)

> Write a small typed Python module that wraps a SQLite table `articles` with
> CRUD operations. Use dataclasses or Pydantic, type hints everywhere, and
> raise specific exceptions (e.g. `ArticleNotFound`).

## 7. Debugging concurrency

> The following code sometimes deadlocks. Explain why and rewrite it so it
> cannot deadlock.
>
> ```python
> import threading
> lock_a = threading.Lock()
> lock_b = threading.Lock()
>
> def t1():
>     lock_a.acquire(); lock_b.acquire()
>     ... ; lock_b.release(); lock_a.release()
>
> def t2():
>     lock_b.acquire(); lock_a.acquire()
>     ... ; lock_a.release(); lock_b.release()
> ```

## 8. Design / architecture

> Design a rate limiter as a small Python library. Support at least two
> algorithms (token bucket, sliding window), be thread-safe, and document the
> public API. No external services.

## 9. React + TypeScript component

> Write a React + TypeScript component `SearchList` that fetches a list of
> items from `/api/items?q=`, debounces the input by 300 ms, and shows
> loading / error / empty states. Use modern hooks and handle race conditions.

## 10. Test-driven utility

> Write a Python function `parse_duration("1h30m") -> timedelta` that accepts
> combinations of `d`, `h`, `m`, `s`. Include a table-driven test suite that
> covers at least six cases, including invalid input.
