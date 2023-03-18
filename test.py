def evaluate(model, x_eval, y_eval, batch_size: int = 200):
    loss, acc = model.evaluate(x_eval, y_eval, batch_size=batch_size)
    return (loss, acc)