def train(args):
  print(args)
  print('Preparing data...')

  tokenizer = AutoTokenizer.from_pretrained(args.model_type)
  dataset = AspectDetectionDataset(
    args.data_dir + '/' + args.dataset + '/' + args.train_file, tokenizer)
  dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=aspect_detection_collate)

  dev_dataset = AspectDetectionDataset(
    args.data_dir + '/' + args.dataset + '/' + args.dev_file, tokenizer, shuffle=False)
  dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=aspect_detection_collate)

  print('Initializing model...')

  model = MIL(args)
  model.cuda()

  #optimizer = torch.optim.Adam(model.parameters())
  optimizer = AdamW(model.parameters(), lr=args.learning_rate)
  scheduler = get_cosine_schedule_with_warmup(
    optimizer, args.no_warmup_steps, args.no_train_steps)

  step = 0
  rng = np.random.default_rng()
  if args.load_model is not None:
    print('Loading model...')
    best_point = torch.load(args.load_model)
    model.load_state_dict(best_point['model'])
    optimizer.load_state_dict(best_point['optimizer'])
    scheduler.load_state_dict(best_point['scheduler'])
    step = best_point['step']

  print('Start training...')
  while step < args.no_train_steps:
    losses = []
    for _, (inp_batch, out_batch) in enumerate(tqdm(dataloader)):
      model.train()

      inp_batch = inp_batch.cuda()
      out_batch = out_batch.cuda().float()

      preds = model(inp_batch, out_batch, step=step)
      document_pred = preds['document']
      sentence_pred = preds['sentence']

      loss = preds['loss']
      losses.append(loss.item())
      loss.backward()

      optimizer.step()
      scheduler.step()

      step += 1
      if step % args.check_every == 0:
        print('Step %d Train Loss %.4f' % (step, np.mean(losses)))

        doc_counts = [[0] * 2] * args.num_aspects
        sent_counts = [[0] * 2] * args.num_aspects

        dev_loss = []
        for _, (inp_batch, out_batch) in enumerate(tqdm(dev_dataloader)):
          model.eval()

          inp_batch = inp_batch.cuda()
          out_batch = out_batch.cuda().float()

          preds = model(inp_batch, out_batch)
          document_pred = preds['document']
          sentence_pred = preds['sentence']

          for bid in range(len(out_batch)):
            for aid in range(args.num_aspects):
              _update_counts(out_batch[bid][aid], document_pred[bid][aid], doc_counts[aid])

              for sid in range(len(sentence_pred[bid])):
                _update_counts(out_batch[bid][aid], sentence_pred[bid][sid][aid], sent_counts[aid])


          loss = preds['loss']
          dev_loss.append(loss.item())

        print('Dev Loss %.4f' % np.mean(dev_loss))

        doc_f1 = []
        sent_f1 = []
        for aid in range(args.num_aspects):
          doc_f1.append(2*doc_counts[aid][0] / float(2*doc_counts[aid][0] + doc_counts[aid][1]))
          sent_f1.append(2*sent_counts[aid][0] / float(2*sent_counts[aid][0] + sent_counts[aid][1]))
        doc_f1 = np.mean(doc_f1) * 100
        sent_f1 = np.mean(sent_f1) * 100

        print('Document F1 %.4f' % doc_f1)
        print('Sentence F1 %.4f' % sent_f1)

        inp = inp_batch[0]
        print('Document prediction', document_pred[0].tolist())
        print('Gold', out_batch[0].tolist())
        print()
        for sid, sentence in enumerate(inp):
          sentence = tokenizer.decode(sentence, skip_special_tokens=True)
          if len(sentence.strip()) == 0:
            continue
          print('Sentence', sid, ':', sentence)
          print(sentence_pred[0][sid].tolist())
        print()


      if step % args.ckpt_every == 0:
        print('Saving...')
        os.makedirs(args.model_dir + '/' + args.dataset, exist_ok=True)
        torch.save({
          'model': model.state_dict(),
          'optimizer': optimizer.state_dict(),
          'scheduler': scheduler.state_dict(),
          'step': step,
          'loss': np.mean(dev_loss)
        }, args.model_dir + '/' + args.dataset + '/' + args.model_name + '.%d.%.2f.%.2f.%.2f' % (step, np.mean(losses), doc_f1, sent_f1))
        losses = []

      if step == args.no_train_steps:
        break